import os
from collections import deque
from copy import deepcopy

from scipy import stats
import torch
from agent.Agent import Agent
from agent.ExchangeAgent import ExchangeAgent
from agent.TradingAgent import TradingAgent
from torch import nn
from torch.nn import functional as F
from util.order.LimitOrder import LimitOrder
from util.util import log_print
from message.Message import Message

from math import sqrt
import numpy as np
import pandas as pd
import datetime

from ABIDES.util.order.MarketOrder import MarketOrder
from utils.utils_data import reset_indexes, normalize_messages, one_hot_encoding_type, to_sparse_representation, tanh_encoding_type
import constants as cst

class WorldAgent(Agent):
    # the objective of this world agent is to replicate the market for the first 30mins and then
    # generated new orderr with the help of a diffusion model for the rest of the day,
    # the diffusion model takes in input the last orders or the last snapshot of the order book
    # and generates new orders for the next time step


    def __init__(self, id, name, type, symbol, date, date_trading_days, model, data_dir, log_orders=True, random_state=None, normalization_terms=None,
                 using_diffusion=False, chosen_model=None, seq_len=256, cond_seq_size=255, cond_type='full', size_type_emb=3, gen_seq_size=1,
                 fix_time=False, type_decode='l1', fix_cancel_bind=False, fix_lob_pad=False, drop_type2_cond=False,
                 depth_temp=1.0, depth_reshape=None, size_reshape=None, depth_noise=0.0,
                 dn_target_exec=0.0):

        super().__init__(id, name, type, random_state=random_state, log_to_file=log_orders)
        self.count_neg_size = 0
        self.next_historical_orders_index = 0
        self.lob_snapshots = []
        self.sparse_lob_snapshots = []
        self.symbol = symbol
        self.date = date
        self.gen_seq_size = gen_seq_size
        self.size_type_emb = size_type_emb
        self.log_orders = log_orders
        self.executed_trades = dict()
        self.state = 'AWAITING_WAKEUP'
        self.model = model
        self.historical_orders, self.historical_lob = self._load_orders_lob(self.symbol, data_dir, self.date, date_trading_days)
        self.historical_order_ids = self.historical_orders[:, 2]
        self.unused_order_ids = np.setdiff1d(np.arange(0, 99999999), self.historical_order_ids)
        self.next_orders = None
        self.subscription_requested = False
        self.date_trading_days = date_trading_days
        self.first_wakeup = True
        self.active_limit_orders = {}
        self.placed_orders = []
        self.count_diff_placed_orders = 0
        self.count_modify = 0
        self.cond_type = cond_type
        self.cond_seq_size = cond_seq_size
        self.seq_len = seq_len
        self.first_generation = True
        self.normalization_terms = normalization_terms
        self.ignored_cancel = 0
        self.generated_orders_out_of_depth = 0
        self.generated_cancel_orders_empty_depth = 0
        self.diff_limit_order_placed = 0
        self.diff_market_order_placed = 0
        self.diff_cancel_order_placed = 0
        self.depth_rounding = 0
        self.last_bid_price = 0
        self.last_ask_price = 0
        # ── Hypothesis-testing flags (all default off = original behavior) ──
        self.fix_time = fix_time                  # H1: feed generated inter-arrivals back into conditioning
        self.type_decode = type_decode            # H2: 'l1' (original), 'l2', or 'prior' type-embedding decode
        self.fix_cancel_bind = fix_cancel_bind    # H3: bind cancels to nearest same-side order instead of dropping
        self.fix_lob_pad = fix_lob_pad            # H5: sentinel-pad empty LOB levels pre-z-score (match training)
        self.drop_type2_cond = drop_type2_cond    # H7: exclude partial cancels from conditioning (match training)
        self.depth_temp = depth_temp              # scale decoded depth z (kappa>1 restores the marketable tail)
        # ── Quantile reshape (decode-time distribution repair; no retrain, no sampler change) ──
        # WHY: the unclamp retrain fixed the SIGN axis of the depth collapse (negative-depth
        # decodes rose 11x on ckpt 0.635) but not the MAGNITUDE axis — DDIM10's B_crossing_limit
        # stayed exactly 0 because deterministic sampling collapses depth's variance, so the rare
        # negative excursions are too small to exceed the current spread. Quantile matching maps
        # each raw z through its rank within the model's OWN recent output distribution (a rolling
        # buffer — self-calibrating in closed loop), then reads that same quantile off the REAL
        # empirical marginal (scripts/build_quantile_targets.py). Unlike --depth-temp (linear scale
        # of a spike => cliff), this is a nonlinear map that uses the continuous intra-spike
        # variation as ranking signal: the bottom ~0.9% of ranks land on genuine -1..-10 tick
        # crossing depths with REAL magnitudes. Midrank tie-handling makes it fail-safe: a fully
        # degenerate source (all z equal) maps to the target's MEDIAN (depth 0), not an extreme.
        self.depth_noise = depth_noise            # per-sample N(0,sigma) on z_depth at decode (LIMIT only)
        self._reshape_warmup = 300                # plain decode until the source buffer has this many z's
        self.depth_reshape_target = None          # sorted real signed-depth array (LIMIT events only)
        if depth_reshape:
            self.depth_reshape_target = np.load(depth_reshape).astype(np.float64)
            self._depth_qgrid = np.linspace(0.0, 1.0, self.depth_reshape_target.size)
            self._z_depth_buf = deque(maxlen=2000)
            print(f"[WorldAgent] depth reshape ON: target={depth_reshape} n={self.depth_reshape_target.size} "
                  f"neg={(self.depth_reshape_target < 0).mean():.3%}")
        self.size_reshape_targets = None          # per-decoded-type sorted real size arrays, [0,1000]
        if size_reshape:
            self.size_reshape_targets = {}
            for k in ("limit", "cancel", "market"):
                t = np.load(os.path.join(size_reshape, f"real_size_{k}.npy")).astype(np.float64)
                self.size_reshape_targets[k] = (t, np.linspace(0.0, 1.0, t.size))
            self._z_size_buf = deque(maxlen=2000)
            print(f"[WorldAgent] size reshape ON: targets from {size_reshape}")
        self.reshape_counts = {"depth_applied": 0, "depth_warmup": 0, "size_applied": 0, "size_warmup": 0}
        # ── Execution-rate feedback controller for depth-noise σ ──
        # WHY: fixed σ=0.3 is realistic over 30 min but over 75 min the drift profile shows a
        # liquidity DEATH SPIRAL: exec% runs 8-15% vs real's 4-6% from the moment generation takes
        # over (while producing ~1.75x fewer events/min than real), the net drain empties the book
        # at ~min 45 — events/bucket collapse 8k→219, spread explodes 1→41 ticks, mid teleports
        # -5%. Cause precedes effect: over-execution, not direction (the sell-pressure direction
        # matched real's). Fix: proportional control of σ to hold the realized exec share at a
        # target. σ_eff = σ_base · clip(target/realized, 0.25, 4.0), realized measured over the
        # last 1000 placed orders (exec = Channel A market orders + Channel B crossing limits,
        # known at placement time). Self-recovering by construction: a post-collapse exec spike
        # (25-38% observed) slams σ down → pure passive replenishment → the book refills.
        self.dn_target_exec = dn_target_exec      # 0 = off (fixed σ, original behavior)
        self._exec_outcomes = deque(maxlen=1000)  # 1 = placed order will execute, 0 = passive
        self._dn_sigma_eff = depth_noise          # last effective σ (for DIAG)
        # PRICE_REANCHOR: same day-open-mid anchor convention as training preprocessing (shared
        # helper — cannot diverge). Applied to conditioning prices only, never to stored
        # snapshots or placed orders. Removes the z≈−4σ price-OOD cliff where the model
        # degenerates (see constants.py). A model trained with the flag REQUIRES it at sim time.
        self.price_anchor = 0.0
        if cst.PRICE_REANCHOR:
            from utils.utils_data import compute_price_anchor
            self.price_anchor = float(compute_price_anchor(self.historical_lob))
            print(f"[WorldAgent] PRICE_REANCHOR on: day anchor = {self.price_anchor} (raw units)")
        # log class priors [limit, cancel, market] for the 'prior' decode. Taken from the
        # real test-set next-event marginals (limit=0.49, cancel=0.48, market=0.03).
        self._type_log_prior = torch.log(torch.tensor([0.49, 0.48, 0.03], device=cst.DEVICE))
        # ── Diagnostics (always on, cheap counters) ──
        self.decoded_type_counts = {1: 0, 3: 0, 4: 0}  # pre-drop decode histogram (1=limit,3=cancel,4=market)
        self.depth_hist = {"neg": 0, "0": 0, "1-2": 0, "3-5": 0, "6+": 0}  # pre-drop generated-depth histogram
        # pre-drop generated-size histogram + running mean/std, split by decoded type (limit/market) —
        # tests whether SIZE is a second, independent collapse axis compounding the "wall" symptom
        # alongside depth: even a correct rate of spread-crossing market orders can't move price if
        # resting size at the touch is systematically oversized/collapsed to a narrow high band.
        _sz_buckets = {"0-50": 0, "51-200": 0, "201-500": 0, "501-1000": 0, ">1000": 0, "neg": 0}
        self.size_hist = {"limit": dict(_sz_buckets), "cancel": dict(_sz_buckets), "market": dict(_sz_buckets)}
        self.size_stats = {"limit": [0.0, 0.0, 0], "cancel": [0.0, 0.0, 0], "market": [0.0, 0.0, 0]}  # [sum, sumsq, n]
        # SEPARATE valid-range-only accumulator (0<=size<=1000, i.e. what survives the drop filter).
        # The full-population size_stats mean/std above is contaminated by the ~30% negative-size
        # decode population — not representative of what actually gets placed into the book. This
        # is the number that's actually comparable across samplers for the wall-mechanism question.
        self.size_stats_valid = {"limit": [0.0, 0.0, 0], "cancel": [0.0, 0.0, 0], "market": [0.0, 0.0, 0]}
        # Channel A vs Channel B: type==4 ("market") always executes via placeMarketOrder,
        # bypassing depth entirely. type==1 ("limit") only executes if its price genuinely
        # crosses the CURRENT opposite-side best price, via the exchange's real matching engine
        # (OrderBook.isMatch/executeOrder). Depth-sign fixes (unclamp/quantile-match) only ever
        # strengthen Channel B; they do nothing for Channel A. Measuring the current split tells
        # us which channel the execution-rate shortfall (real ~7.0% vs deterministic ~3.7-5.0%)
        # is actually coming from, before investing more effort in reshaping depth specifically.
        self.channel_b_would_cross = 0    # decoded type==1 orders whose price crosses the book NOW
        self.drop_counts = {"size_range": 0, "limit_out_of_depth": 0,
                            "cancel_no_best": 0, "cancel_side_empty": 0}
        self.resample_total_batches = 0
        self.resample_extra_batches = 0
        self.resample_exhausted = 0   # times the resample cap was hit and the wakeup was rescheduled
        self.cond_stats = {}                      # per-feature [min, max, sum, count] of z-scored conditioning
        self.using_diffusion = using_diffusion
        self.chosen_model = chosen_model
        if using_diffusion:
            self.starting_time_diffusion = '15min'
        else:
            self.starting_time_diffusion = '157780min'

    def kernelStarting(self, startTime):
        # self.kernel is set in Agent.kernelInitializing()
        super().kernelStarting(startTime)
        self.oracle = self.kernel.oracle
        self.exchangeID = self.kernel.findAgentByType(ExchangeAgent)
        self.mkt_open = startTime

    def kernelTerminating(self):
        # self.kernel is set in Agent.kernelInitializing()
        super().kernelTerminating()
        print("World Agent terminating.")
        print("World Agent ignored {} cancel orders".format(self.ignored_cancel))
        print("=== WORLDAGENT DIAGNOSTICS ===")
        print("DIAG decoded_pre_drop: limit={} cancel={} market={}".format(
            self.decoded_type_counts[1], self.decoded_type_counts[3], self.decoded_type_counts[4]))
        print("DIAG placed: limit={} cancel={} market={}".format(
            self.diff_limit_order_placed, self.diff_cancel_order_placed, self.diff_market_order_placed))
        print("DIAG execution_channels: A_market_order={}  B_crossing_limit={}  (A bypasses depth entirely "
              "via placeMarketOrder; B only fires if a decoded LIMIT price crosses the book NOW, via the "
              "exchange's real matching engine — this is the only channel depth-sign fixes can affect)".format(
                  self.diff_market_order_placed, self.channel_b_would_cross))
        print("DIAG drops: size_range={} limit_out_of_depth={} cancel_no_best={} cancel_side_empty={}".format(
            self.drop_counts["size_range"], self.drop_counts["limit_out_of_depth"],
            self.drop_counts["cancel_no_best"], self.drop_counts["cancel_side_empty"]))
        print("DIAG cancel_empty_depth_fallbacks={} ignored_cancel_at_place={}".format(
            self.generated_cancel_orders_empty_depth, self.ignored_cancel))
        print("DIAG resample: total_batches={} extra_batches={} exhausted={}".format(
            self.resample_total_batches, self.resample_extra_batches, self.resample_exhausted))
        print("DIAG depth_pre_drop: neg={} 0={} 1-2={} 3-5={} 6+={}".format(
            self.depth_hist["neg"], self.depth_hist["0"], self.depth_hist["1-2"],
            self.depth_hist["3-5"], self.depth_hist["6+"]))
        if self.depth_reshape_target is not None or self.size_reshape_targets is not None or self.depth_noise > 0:
            print("DIAG reshape: depth_applied={} depth_warmup={} size_applied={} size_warmup={} depth_noise={}".format(
                self.reshape_counts["depth_applied"], self.reshape_counts["depth_warmup"],
                self.reshape_counts["size_applied"], self.reshape_counts["size_warmup"], self.depth_noise))
        if self.dn_target_exec > 0.0:
            realized = (sum(self._exec_outcomes) / len(self._exec_outcomes)) if self._exec_outcomes else float("nan")
            print("DIAG dn_controller: target_exec={} realized_exec_last{}={:.3f} sigma_base={} sigma_eff_final={:.3f}".format(
                self.dn_target_exec, self._exec_outcomes.maxlen, realized, self.depth_noise, self._dn_sigma_eff))
        for k in ("limit", "cancel", "market"):
            h = self.size_hist[k]
            s = self.size_stats[k]
            mean = s[0] / s[2] if s[2] else float("nan")
            std = ((s[1] / s[2]) - mean ** 2) ** 0.5 if s[2] else float("nan")
            v = self.size_stats_valid[k]
            vmean = v[0] / v[2] if v[2] else float("nan")
            vstd = ((v[1] / v[2]) - vmean ** 2) ** 0.5 if v[2] else float("nan")
            print("DIAG size_pre_drop[{}]: neg={} 0-50={} 51-200={} 201-500={} 501-1000={} >1000={}  "
                  "mean={:.1f} std={:.1f} n={}  |  valid-range-only: mean={:.1f} std={:.1f} n={}".format(
                      k, h["neg"], h["0-50"], h["51-200"], h["201-500"], h["501-1000"], h[">1000"],
                      mean, std, s[2], vmean, vstd, v[2]))
        for feat, s in sorted(self.cond_stats.items()):
            mean = s[2] / s[3] if s[3] else float("nan")
            print("DIAG cond_z[{}]: min={:.2f} mean={:.2f} max={:.2f} n={}".format(feat, s[0], mean, s[1], s[3]))
        print("=== END DIAGNOSTICS ===")

    def requestDataSubscription(self, symbol, levels):
        self.sendMessage(recipientID=self.exchangeID,
                         msg=Message({"msg": "MARKET_DATA_SUBSCRIPTION_REQUEST",
                                      "sender": self.id,
                                      "symbol": symbol,
                                      "levels": levels,
                                      "freq": 0})  # if freq is 0 all the LOB updates will be provided
                         )
        
    def cancelDataSubscription(self):
        self.sendMessage(recipientID=self.exchangeID,
                         msg=Message({"msg": "CANCEL_MARKET_DATA_SUBSCRIPTION",
                                      "sender": self.id,
                                      "symbol": self.symbol})
                         )

    def wakeup(self, currentTime):
        super().wakeup(currentTime)
        #make a print every 5 minutes
        
        if currentTime.minute % 5 == 0 and currentTime.second == 00:
            print("Current time: {}".format(currentTime))
            #print("Number of generated orders out of depth: {}".format(self.generated_orders_out_of_depth))
            #print("Number of generated cancel orders unmatched: {}".format(self.generated_cancel_orders_empty_depth))
            #print("Number of generated cancel orders matched: {}".format(self.diff_cancel_order_placed))
            #print("Number of negative size: {}".format(self.count_neg_size))
            #print("Number of generated placed orders: {}".format(self.count_diff_placed_orders))
            #print("Of which {} market order and {} limit order".format(self.diff_market_order_placed, self.diff_limit_order_placed))
            now = datetime.datetime.now()
            current_time = now.strftime("%H:%M:%S")
            #print("Current Time =", current_time)

        if self.first_wakeup:
            self.state = 'PRE_GENERATING'
            offset = datetime.timedelta(seconds=self.historical_orders[0, 0])
            time_next_wakeup = currentTime + offset
            self.setWakeup(time_next_wakeup)
            self.requestDataSubscription(self.symbol, levels=10)
            self.first_wakeup = False

        # if current time is between 09:30 and 09:45, then we are in the pre-open phase
        elif self.mkt_open <= currentTime <= self.mkt_open + pd.Timedelta(self.starting_time_diffusion):
            next_order = self.historical_orders[self.next_historical_orders_index]
            self.last_offset_time = next_order[0]
            self.placeOrder(currentTime, next_order)
            self.next_historical_orders_index += 1
            if self.next_historical_orders_index < len(self.historical_orders):
                offset = datetime.timedelta(seconds=self.historical_orders[self.next_historical_orders_index, 0])
                self.setWakeup(currentTime + offset + datetime.timedelta(microseconds=1))
            else:
                return
            
        elif currentTime > self.mkt_open + pd.Timedelta(self.starting_time_diffusion) and not self.using_diffusion:
            print("cancelling data subscription")
            self.cancelDataSubscription()
            
        elif currentTime > self.mkt_open + pd.Timedelta(self.starting_time_diffusion) and self.using_diffusion:
            self.state = 'GENERATING'
            # we generate the first order then the others will be generated everytime we receive the update of the lob
            if self.first_generation:
                if self.chosen_model == 'CGAN':        
                    # we need to fit the temporal distance with a gamma distribution
                    temporal_distance = self.historical_orders[:, 0]
                    # remove all the zeros from the temporal distance
                    temporal_distance = temporal_distance[temporal_distance > 0]
                    self.shape_temp_distance, self.loc_temp_distance, self.scale_temp_distance = stats.gamma.fit(temporal_distance)
        
                generated_orders = self._generate_order(currentTime)
                if len(generated_orders) == 0:
                    # resample loop exhausted — retry shortly with fresh conditioning
                    self.setWakeup(currentTime + datetime.timedelta(seconds=1))
                    return
                self.next_orders = generated_orders
                self.first_generation = False
                offset_time = datetime.timedelta(seconds=generated_orders[0][0])
                self.setWakeup(currentTime + offset_time + datetime.timedelta(microseconds=1))
                return

            # Place the next queued order, or generate a new batch if the queue is empty.
            # The O(n) kernel-queue scan was removed — ABIDES event ordering guarantees that
            # exchange responses (ORDER_ACCEPTED etc.) are delivered via receiveMessage before
            # the next wakeup fires, so waiting is unnecessary and was causing quadratic slowdown.
            if len(self.next_orders) > 0:
                if self.fix_time:
                    # H1: stamp the conditioning history with this order's generated
                    # inter-arrival. Without this, last_offset_time keeps the final
                    # historical order's gap forever and the time channel of the
                    # conditioning is a frozen constant during generation.
                    self.last_offset_time = self.next_orders[0][0]
                self.placeOrder(currentTime, self.next_orders[0])
                offset_time = datetime.timedelta(seconds=self.next_orders[0][0])
                self.next_orders = self.next_orders[1:]
            else:
                self.next_orders = self._generate_order(currentTime)
                if len(self.next_orders) == 0:
                    # resample loop exhausted — retry shortly with fresh conditioning
                    self.setWakeup(currentTime + datetime.timedelta(seconds=1))
                    return
                offset_time = datetime.timedelta(seconds=self.next_orders[0][0])
            self.setWakeup(currentTime + offset_time + datetime.timedelta(microseconds=1))
            return
            
            

        
    def receiveMessage(self, currentTime, msg):
        if currentTime > self.mkt_open + pd.Timedelta(self.starting_time_diffusion) and not self.using_diffusion:
            return
        
        super().receiveMessage(currentTime, msg)
        if msg.body['msg'] == 'MARKET_DATA':
            self._update_lob_snapshot(msg)
            self._update_active_limit_orders()

        # if we had placed a market order and it is executed we receive the message of the limit order filled, so if it was a buy we receive a sell
        elif msg.body['msg'] == 'ORDER_EXECUTED':
            direction = 1 if msg.body['order'].is_buy_order else -1
            self.placed_orders.append(np.array([self.last_offset_time, 4, msg.body['order'].order_id, msg.body['order'].quantity, msg.body['order'].limit_price, direction]))
            if len(self.placed_orders) > self.seq_len * 2:
                self.placed_orders = self.placed_orders[-self.seq_len * 2:]
            self.logEvent('ORDER_EXECUTED', msg.body['order'].to_dict())

        elif msg.body['msg'] == 'ORDER_ACCEPTED':
            direction = 1 if msg.body['order'].is_buy_order else -1
            self.placed_orders.append(np.array([self.last_offset_time, 1, msg.body['order'].order_id, msg.body['order'].quantity, msg.body['order'].limit_price, direction]))
            if len(self.placed_orders) > self.seq_len * 2:
                self.placed_orders = self.placed_orders[-self.seq_len * 2:]
            self.logEvent('ORDER_ACCEPTED', msg.body['order'].to_dict())

        elif msg.body['msg'] == 'ORDER_CANCELLED':
            direction = 1 if msg.body['order'].is_buy_order else -1
            self.placed_orders.append(np.array([self.last_offset_time, 3, msg.body['order'].order_id, msg.body['order'].quantity, msg.body['order'].limit_price, direction]))
            if len(self.placed_orders) > self.seq_len * 2:
                self.placed_orders = self.placed_orders[-self.seq_len * 2:]
            self.logEvent('ORDER_CANCELLED', msg.body['order'].to_dict())

    def placeOrder(self, currentTime, order):
        order_id = order[2]
        type = order[1]
        quantity = order[3]
        price = int(order[4])
        direction = order[5]
        if quantity > 0:
            direction = False if direction == -1 else True
            if type == 1:
                self.placeLimitOrder(self.symbol, quantity, is_buy_order=direction, limit_price=price, order_id=order_id)

            elif type == 2 or type == 3:
                if order_id in self.active_limit_orders:
                    old_order = self.active_limit_orders[order_id]
                    del self.active_limit_orders[order_id]
                else:
                    self.ignored_cancel += 1
                    return
                    # raise Exception("trying to cancel an order that doesn't exist")
                if type == 3:
                    # total deletion of a limit order
                    self.cancelOrder(old_order)
                elif type == 2:
                    # partial deletion of a limit order
                    new_order = LimitOrder(
                        agent_id=self.id, 
                        time_placed=self.currentTime, 
                        symbol=self.symbol, 
                        quantity=old_order.quantity-quantity, 
                        is_buy_order=old_order.is_buy_order, 
                        limit_price=old_order.limit_price, 
                        order_id=old_order.order_id, 
                        tag=None
                    )
                    self.modifyOrder(old_order, new_order)
                    # H7: training data drops ALL type-2 (partial cancel) rows before
                    # windowing, so conditioning windows never contained them. With the
                    # flag on, keep them out of the sim conditioning history too.
                    if not self.drop_type2_cond:
                        self.placed_orders.append(np.array([order[0], 2, new_order.order_id, quantity, new_order.limit_price, direction]))
                        if len(self.placed_orders) > self.seq_len * 2:
                            self.placed_orders = self.placed_orders[-self.seq_len * 2:]

            elif type == 4:
                # if type == 4 it means that it is an execution order, so if it is an execution order of a sell limit order
                # we place a buy market order of the same quantity and viceversa
                is_buy_order = False if direction else True
                # the current order_id is the order_id of the sell (buy) limit order filled, 
                # so we need to assign to the market order another order_id
                order_id = self.unused_order_ids[0]
                self.unused_order_ids = self.unused_order_ids[1:]
                self.placeMarketOrder(self.symbol, quantity, is_buy_order=is_buy_order, order_id=order_id)
        else:
            log_print("Agent ignored order of quantity zero: {}", order)

    def _generate_order(self, currentTime, max_attempts=100):
        generated = None
        post_processed_orders = []
        attempts = 0
        # Cap the resample loop: each iteration is a full diffusion sample(). If every
        # generated order in a batch keeps hitting a drop filter (e.g. a checkpoint that
        # reliably emits out-of-range sizes for the current conditioning state), the loop
        # would otherwise spin forever and hang the whole simulation. On exhaustion we
        # return an empty list; the caller reschedules a retry wakeup so fresh market data
        # can shift the conditioning out of the degenerate state.
        while len(post_processed_orders) == 0 and attempts < max_attempts:
            attempts += 1
            self.resample_total_batches += 1
            if attempts > 1:
                # entire previous batch was dropped by postprocess filters — resampling
                self.resample_extra_batches += 1
            if self.chosen_model == 'TRADES':
                if self.cond_type == 'full':
                    orders = np.array(self.placed_orders[-self.cond_seq_size:])
                    cond_orders = self._preprocess_orders_for_diff_cond(orders, np.array(self.lob_snapshots[-self.cond_seq_size -1:]))
                    lob_snapshots = np.array(self.lob_snapshots[-self.cond_seq_size-1:])
                    cond_lob = torch.from_numpy(self._z_score_orderbook(lob_snapshots)).to(cst.DEVICE, torch.float32)
                    cond_lob = cond_lob.unsqueeze(0)
                elif self.cond_type == 'only_event':
                    orders = np.array(self.placed_orders[-self.cond_seq_size:])
                    cond_orders = self._preprocess_orders_for_diff_cond(orders, np.array(self.lob_snapshots[-self.cond_seq_size -1:]))
                    cond_lob = None
                else:
                    raise ValueError("cond_type not recognized")
                cond_orders = cond_orders.unsqueeze(0)   
                x = torch.zeros(1, self.gen_seq_size, cst.LEN_ORDER, device=cst.DEVICE, dtype=torch.float32)
                generated = self.model.sample(cond_orders=cond_orders, x=x, cond_lob=cond_lob)
                post_processed_orders = []
                for i in range(generated.shape[1]):
                    order = self._postprocess_generated_TRADES(generated[0, i, :])
                    if order is not None:
                        post_processed_orders.append(order)
                
            elif self.chosen_model == 'CGAN':
                cond_market_features = self._preprocess_market_features_for_cgan(np.array(self.lob_snapshots[-(self.seq_len)*2+1:]))
                '''
                 cond_market_features = 
                    ['volume_imbalance_1',
                    'volume_imbalance_5', 
                    'absolute_volume_1', 
                    'absolute_volume_5', 
                    'spread', 
                    'order_sign_imbalance_256', 
                    'order_sign_imbalance_128', 
                    'returns_1', 
                    'returns_50']
                '''
                noise = torch.randn(1, 1, self.model.generator_lstm_hidden_state_dim).to(cst.DEVICE, torch.float32)
                generated = self.model.sample(noise=noise, cond_market_features=cond_market_features)
                generated = self.model.post_process_order(generated)
                # generated = ['event_type', 'size', 'direction', 'depth', 'cancel_depth', 'quantity_100', 'quantity_type']
                generated = generated[0, 0, :]
                generated = self._postprocess_generated_gan(generated)
                if generated is not None:
                    post_processed_orders = [generated]
                    # generated = [offset, order_type, order_id, size, price, direction]
        if len(post_processed_orders) == 0:
            self.resample_exhausted += 1
        return post_processed_orders


    def placeLimitOrder(self, symbol, quantity, is_buy_order, limit_price, order_id=None, ignore_risk=True, tag=None):
        order = LimitOrder(self.id, self.currentTime, symbol, quantity, is_buy_order, limit_price, order_id, tag)
        self.sendMessage(self.exchangeID, Message({"msg": "LIMIT_ORDER", "sender": self.id, "order": order}))
        # Log this activity.
        if self.log_orders: self.logEvent('ORDER_SUBMITTED', order.to_dict())

    def placeMarketOrder(self, symbol, quantity, is_buy_order, order_id=None, ignore_risk=True, tag=None):
        """
          The market order is created as multiple limit orders crossing the spread walking the book until all the quantities are matched.
        """
        order = MarketOrder(self.id, self.currentTime, symbol, quantity, is_buy_order, order_id)
        self.sendMessage(self.exchangeID, Message({"msg": "MARKET_ORDER", "sender": self.id, "order": order}))
        if self.log_orders: self.logEvent('ORDER_SUBMITTED', order.to_dict())

    def cancelOrder(self, order):
        """Used by any Trading Agent subclass to cancel any order.
        The order must currently appear in the agent's open orders list."""
        if isinstance(order, LimitOrder):
            self.sendMessage(self.exchangeID, Message({"msg": "CANCEL_ORDER", "sender": self.id,
                                                       "order": order}))
            # Log this activity.
            if self.log_orders: self.logEvent('CANCEL_SUBMITTED', order.to_dict())
        else:
            log_print("order {} of type, {} cannot be cancelled", order, type(order))

    def modifyOrder(self, order, newOrder):
        """ Used by any Trading Agent subclass to modify any existing limit order.  The order must currently
            appear in the agent's open orders list.  Some additional tests might be useful here
            to ensure the old and new orders are the same in some way."""
        self.sendMessage(self.exchangeID, Message({"msg": "MODIFY_ORDER", "sender": self.id,
                                                   "order": order, "new_order": newOrder}))
        # Log this activity.
        if self.log_orders: self.logEvent('MODIFY_ORDER', order.to_dict())

    def _postprocess_generated_gan(self, generated):
        ''' we need to go from the output of the cgan model to an actual order '''
        generated = generated.cpu().detach().numpy()
        # firstly we generate the offset 
        offset = stats.gamma.rvs(self.shape_temp_distance, self.loc_temp_distance, self.scale_temp_distance)
        
        direction = generated[2]
        quantity_type = generated[6]
        order_type = generated[0]
        # order type == -1 -> limit order
        # order type == 0 -> cancel order
        # order type == 1 -> market order
        order_type += 2
        if order_type == 3 or order_type == 2:
            order_type += 1
        # order type == 1 -> limit order
        # order type == 3 -> cancel order
        # order type == 4 -> market order
        
        # we return the depth, the cancel depth, the size and the quantity100 to the original scale
        mean_depth = self.normalization_terms["lob"][12]
        std_depth = self.normalization_terms["lob"][13]
        mean_cancel_depth = self.normalization_terms["lob"][8]
        std_cancel_depth = self.normalization_terms["lob"][9]
        mean_size_100 = self.normalization_terms["lob"][10]
        std_size_100 = self.normalization_terms["lob"][11]
        mean_size = self.normalization_terms["lob"][14]
        std_size = self.normalization_terms["lob"][15]
        depth = int(generated[3] * std_depth + mean_depth)
        cancel_depth = int(generated[4] * std_cancel_depth + mean_cancel_depth)
        # we are considering only the first 10 levels of the order book so we need to check if the cancel depth is greater than 9
        if cancel_depth > 9:
            return None
        size_100 = generated[5] * std_size_100 + mean_size_100
        size = int(generated[1] * std_size + mean_size)
        
        if quantity_type == -1:
            size = int(size_100)*100
            
        if order_type == 1:
            order_id = self.unused_order_ids[0]
            self.unused_order_ids = self.unused_order_ids[1:]
            if direction == 1:
                bid_side = self.lob_snapshots[-1][2::4]
                bid_price = bid_side[0]
                if bid_price == 0:
                    bid_price = self.last_bid_price
                else:
                    self.last_bid_price = bid_price
                last_price = bid_side[-1] 
                price = bid_price - depth*100
                # if the first 10 levels are full and the price is less than the last price we generate another order
                # because we consider only the first 10 levels
                if price < last_price and last_price > 0:
                    self.generated_orders_out_of_depth += 1
                    return None
                self.diff_limit_order_placed += 1
            else:
                ask_side = self.lob_snapshots[-1][0::4]
                ask_price = ask_side[0]
                if ask_price == 0:
                    ask_price = self.last_ask_price
                else:
                    self.last_ask_price = ask_price
                last_price = ask_side[-1]
                price = ask_price + depth*100
                if price > last_price and last_price > 0:
                    self.generated_orders_out_of_depth += 1
                    return None
                self.diff_limit_order_placed += 1

        elif order_type == 3:
            if direction == 1:
                bid_side = self.lob_snapshots[-1][2::4]
                bid_price = bid_side[0]
                if bid_price == 0:
                    return None
                else:
                    self.last_bid_price = bid_price
                #select the price at depth = cancel_depth
                price = bid_side[cancel_depth]
                # search all the active limit orders with the same price
                orders_with_same_price = [order for order in self.active_limit_orders.values() if order.limit_price == price]
                # if there are no orders with the same price then we generate another order
                if len(orders_with_same_price) == 0:
                    self.generated_cancel_orders_empty_depth += 1
                    #chech if there are buy limit orders active
                    if len([order for order in self.active_limit_orders.values() if order.is_buy_order]) == 0:
                        return None
                    # find the order with the closest price and quantity
                    order_id = min(self.active_limit_orders.values(), key=lambda x: (abs(x.limit_price - price), abs(x.quantity - size))).order_id
                else:
                    # we select the order with the quantity closer to the quantity generated
                    order_id = min(orders_with_same_price, key=lambda x: abs(x.quantity - size)).order_id
                    self.diff_cancel_order_placed += 1

            else:
                ask_side = self.lob_snapshots[-1][0::4]
                ask_price = ask_side[0]
                if ask_price == 0:
                    return None
                else:
                    self.last_ask_price = ask_price
                price = ask_side[cancel_depth]
                # search all the active limit orders in the same level
                orders_with_same_price = [order for order in self.active_limit_orders.values() if order.limit_price == price]
                # if there are no orders with the same price then we generate another order
                if len(orders_with_same_price) == 0:
                    self.generated_cancel_orders_empty_depth += 1
                    #chech if there are sell limit orders active
                    if len([order for order in self.active_limit_orders.values() if not order.is_buy_order]) == 0:
                        return None
                    order_id = min(self.active_limit_orders.values(), key=lambda x: (abs(x.limit_price - price), abs(x.quantity - size))).order_id
                else:
                    # we select the order with the quantity near to the quantity generated
                    order_id = min(orders_with_same_price, key=lambda x: abs(x.quantity - size)).order_id
                    self.diff_cancel_order_placed += 1

        elif order_type == 4:
            self.diff_market_order_placed += 1
            if direction == 1:
                price = self.lob_snapshots[-1][0]
            else:
                price = self.lob_snapshots[-1][2]
            order_id = 0
            # the diffusion gives in output market order and not execution of limit order,
            # so we transform market orders in execution orders of the opposite side as the original message files
            direction = -direction
        self.count_diff_placed_orders += 1
        return np.array([offset, order_type, order_id, size, price, direction])
        
        

    @staticmethod
    def _quantile_remap(z, buf, target_sorted, target_qgrid):
        """Online quantile match: rank z within the model's own recent raw outputs (midrank over
        ties), then read the same quantile off the sorted REAL target array. Midrank matters:
        if the source has fully collapsed (all buffered z identical), q=0.5 and the output is the
        target's median — a graceful no-op, not a jump to the distribution's extreme."""
        arr = np.fromiter(buf, dtype=np.float64)
        n = arr.size
        less = np.count_nonzero(arr < z)
        eq = np.count_nonzero(arr == z)
        q = (less + 0.5 * (eq + 1.0)) / (n + 1.0)
        return float(np.interp(q, target_qgrid, target_sorted))

    def _postprocess_generated_TRADES(self, generated):
        ''' we need to go from the output of the diffusion model to an actual order '''
        direction = generated[self.size_type_emb+3]
        if direction < 0:
            direction = -1
        else:
            direction = 1
        
        #order_type = torch.argmax(generated[1:self.size_type_emb+1]).item() + 1
        _type_diffs = self.model.type_embedder.weight.data - generated[1:self.size_type_emb+1]
        if self.type_decode == 'prior':
            # H2 fix: Bayes-corrected nearest-anchor decode. Under a Gaussian likelihood,
            # log p(class | x) ~ -0.5*||x - anchor||^2 + log prior(class). Plain argmin
            # over raw distance ignores the prior, so MARKET's geometrically large decision
            # region wins far more often than its true ~3% rate warrants — that is what
            # blows up to 24% market decode under high-variance sampling and drives price
            # drift. Adding -log(prior) penalizes rare classes by exactly the right amount.
            d2 = torch.sum(_type_diffs ** 2, dim=1)
            score = 0.5 * d2 - self._type_log_prior
            order_type = torch.argmin(score).item() + 1
        elif self.type_decode == 'l2':
            # H2 variant: squared-euclidean nearest anchor instead of L1
            order_type = torch.argmin(torch.sum(_type_diffs ** 2, dim=1)).item() + 1
        else:
            order_type = torch.argmin(torch.sum(torch.abs(_type_diffs), dim=1)).item() + 1

        if order_type == 3 or order_type == 2:
            order_type += 1
        # order type == 1 -> limit order
        # order type == 3 -> cancel order
        # order type == 4 -> market order
        self.decoded_type_counts[order_type] += 1  # pre-drop histogram (model's raw intent)

        # we return the size and the time to the original scale
        z_size_raw = generated[self.size_type_emb+1].item()
        size = round(z_size_raw * self.normalization_terms["event"][1] + self.normalization_terms["event"][0], ndigits=0)

        # Pre-drop size histogram + running mean/std, split limit vs market. Real markets have
        # depth0~58.6% too (concentration at the touch isn't itself abnormal) — the sharper gap is
        # execution rate (real/DDPM ~7-8% vs deterministic samplers ~3.7-5%) and the absolute wall
        # sizes (40k-200k share means vs real ~2-4k). This checks whether SIZE independently
        # collapses to a narrow/high band under deterministic sampling, compounding the wall
        # regardless of depth's sign.
        _sk = {1: "limit", 3: "cancel", 4: "market"}[order_type]
        # SIZE quantile reshape: remap this z's rank (within the model's own recent raw z_size
        # outputs) onto the REAL per-type size marginal. Also eliminates the ~30-40% negative-size
        # decode population entirely (targets live in [0,1000]), so the size_range drop-and-resample
        # waste (42% of batches under DDPM) disappears — a throughput win on top of realism.
        if self.size_reshape_targets is not None:
            self._z_size_buf.append(z_size_raw)
            if len(self._z_size_buf) >= self._reshape_warmup:
                _t, _q = self.size_reshape_targets[_sk]
                size = max(1.0, round(self._quantile_remap(z_size_raw, self._z_size_buf, _t, _q)))
                self.reshape_counts["size_applied"] += 1
            else:
                self.reshape_counts["size_warmup"] += 1
        if size < 0:        self.size_hist[_sk]["neg"] += 1
        elif size <= 50:     self.size_hist[_sk]["0-50"] += 1
        elif size <= 200:    self.size_hist[_sk]["51-200"] += 1
        elif size <= 500:    self.size_hist[_sk]["201-500"] += 1
        elif size <= 1000:   self.size_hist[_sk]["501-1000"] += 1
        else:                self.size_hist[_sk][">1000"] += 1
        self.size_stats[_sk][0] += size; self.size_stats[_sk][1] += size * size; self.size_stats[_sk][2] += 1
        if 0 <= size <= 1000:
            v = self.size_stats_valid[_sk]
            v[0] += size; v[1] += size * size; v[2] += 1
        # depth_temp: scale the decoded depth z-score before denormalizing. Training clamped
        # depth to >=0, so the model piles its depth output near 0 (passive); only sampling
        # variance spilling below 0 produces marketable orders that execute and move price.
        # Deterministic few-step sampling doesn't spill -> freeze. Scaling z_depth by kappa>1
        # widens (and slightly shifts) the depth distribution so some mass crosses into
        # negative (marketable) territory, restoring the tail without a stochastic sampler.
        # kappa=1.0 -> identical to original behavior.
        z_depth_raw = generated[-1].item()
        z_depth = z_depth_raw * self.depth_temp
        # --depth-noise: per-sample N(0,sigma) on the depth channel only, at decode, LIMIT only.
        # The "dumb variance fix" comparator to quantile reshape: unlike CHURN/HYBRID it cannot
        # destabilize other channels (it never enters the sampler or the conditioning), and unlike
        # --depth-temp it acts per-SAMPLE, so it can split the collapsed atom rather than slide it.
        if self.depth_noise > 0.0 and order_type == 1:
            sigma = self.depth_noise
            if self.dn_target_exec > 0.0 and len(self._exec_outcomes) >= 300:
                # proportional controller: throttle σ when the market over-executes, open it
                # back up when it under-executes. Bounds keep it a modulation, not a takeover.
                realized = sum(self._exec_outcomes) / len(self._exec_outcomes)
                sigma = self.depth_noise * min(4.0, max(0.25, self.dn_target_exec / max(realized, 1e-3)))
            self._dn_sigma_eff = sigma
            z_depth = z_depth + sigma * float(np.random.randn())
        depth = round(z_depth * self.normalization_terms["event"][7] + self.normalization_terms["event"][6], ndigits=0)
        # DEPTH quantile reshape (LIMIT only — market bypasses depth at placement; cancel depth is
        # a matching key, not a distributional quantity). Overrides depth-temp/noise when active.
        if self.depth_reshape_target is not None and order_type == 1:
            self._z_depth_buf.append(z_depth_raw)
            if len(self._z_depth_buf) >= self._reshape_warmup:
                depth = round(self._quantile_remap(z_depth_raw, self._z_depth_buf,
                                                   self.depth_reshape_target, self._depth_qgrid))
                self.reshape_counts["depth_applied"] += 1
            else:
                self.reshape_counts["depth_warmup"] += 1
        time = generated[0].item() * self.normalization_terms["event"][5] + self.normalization_terms["event"][4]

        # Pre-drop depth histogram. The freeze is a depth-diversity problem: few-step
        # deterministic sampling collapses depth toward its mean (~1 tick, passive), so
        # orders never cross the spread, never execute, and stack into walls. The negative
        # (marketable) tail is what drives executions and price movement — track it directly.
        if depth < 0:      self.depth_hist["neg"] += 1
        elif depth == 0:   self.depth_hist["0"] += 1
        elif depth <= 2:   self.depth_hist["1-2"] += 1
        elif depth <= 5:   self.depth_hist["3-5"] += 1
        else:              self.depth_hist["6+"] += 1

        # if the price or the size are negative we return None and we generate another order
        if size < 0 or size > 1000:
            self.count_neg_size += 1
            self.drop_counts["size_range"] += 1
            return None

        _will_exec = False   # controller signal: does this placed order consume liquidity NOW?
        
        # if the time is negative we approximate to 1 microsecond
        if time <= 0:
            time = 0.0000001

        if order_type == 1:
            order_id = self.unused_order_ids[0]
            self.unused_order_ids = self.unused_order_ids[1:]
            if direction == 1:
                bid_side = self.lob_snapshots[-1][2::4]
                bid_price = bid_side[0]
                if bid_price == 0:
                    bid_price = self.last_bid_price
                else:
                    self.last_bid_price = bid_price
                last_price = bid_side[-1]
                price = bid_price - depth*100
                # if the first 10 levels are full and the price is less than the last price we generate another order
                # because we consider only the first 10 levels
                if price < last_price and last_price > 0:
                    self.generated_orders_out_of_depth += 1
                    self.drop_counts["limit_out_of_depth"] += 1
                    return None
                self.diff_limit_order_placed += 1
                # Channel B check: a buy limit crosses if priced >= the CURRENT best ask.
                current_ask = self.lob_snapshots[-1][0::4][0]
                if current_ask > 0 and price >= current_ask:
                    self.channel_b_would_cross += 1
                    _will_exec = True
            else:
                ask_side = self.lob_snapshots[-1][0::4]
                ask_price = ask_side[0]
                if ask_price == 0:
                    ask_price = self.last_ask_price
                else:
                    self.last_ask_price = ask_price
                last_price = ask_side[-1]
                price = ask_price + depth*100
                if price > last_price and last_price > 0:
                    self.generated_orders_out_of_depth += 1
                    self.drop_counts["limit_out_of_depth"] += 1
                    return None
                self.diff_limit_order_placed += 1
                # Channel B check: a sell limit crosses if priced <= the CURRENT best bid.
                current_bid = self.lob_snapshots[-1][2::4][0]
                if current_bid > 0 and price <= current_bid:
                    self.channel_b_would_cross += 1
                    _will_exec = True

        elif order_type == 3:
            if direction == 1:
                bid_side = self.lob_snapshots[-1][2::4]
                bid_price = bid_side[0]
                if bid_price == 0:
                    self.drop_counts["cancel_no_best"] += 1
                    return None
                else:
                    self.last_bid_price = bid_price
                price = bid_price - depth*100
                if self.fix_cancel_bind:
                    # H3: always bind to the nearest same-side resting order; only drop
                    # when the side is truly empty. Original behavior drops far more often.
                    same_side = [o for o in self.active_limit_orders.values() if o.is_buy_order]
                    if len(same_side) == 0:
                        self.drop_counts["cancel_side_empty"] += 1
                        return None
                    order_id = min(same_side, key=lambda x: (abs(x.limit_price - price), abs(x.quantity - size))).order_id
                    self.diff_cancel_order_placed += 1
                else:
                    # search all the active limit orders with the same price
                    orders_with_same_price = [order for order in self.active_limit_orders.values() if order.limit_price == price]
                    # if there are no orders with the same price then we generate another order
                    if len(orders_with_same_price) == 0:
                        self.generated_cancel_orders_empty_depth += 1
                        #chech if there are buy limit orders active
                        if len([order for order in self.active_limit_orders.values() if order.is_buy_order]) == 0:
                            self.drop_counts["cancel_side_empty"] += 1
                            return None
                        # find the order with the closest price and quantity
                        order_id = min(self.active_limit_orders.values(), key=lambda x: (abs(x.limit_price - price), abs(x.quantity - size))).order_id
                    else:
                        # we select the order with the quantity closer to the quantity generated
                        order_id = min(orders_with_same_price, key=lambda x: abs(x.quantity - size)).order_id
                        self.diff_cancel_order_placed += 1

            else:
                ask_side = self.lob_snapshots[-1][0::4]
                ask_price = ask_side[0]
                if ask_price == 0:
                    self.drop_counts["cancel_no_best"] += 1
                    return None
                else:
                    self.last_ask_price = ask_price
                price = ask_price + depth*100
                if self.fix_cancel_bind:
                    same_side = [o for o in self.active_limit_orders.values() if not o.is_buy_order]
                    if len(same_side) == 0:
                        self.drop_counts["cancel_side_empty"] += 1
                        return None
                    order_id = min(same_side, key=lambda x: (abs(x.limit_price - price), abs(x.quantity - size))).order_id
                    self.diff_cancel_order_placed += 1
                else:
                    # search all the active limit orders in the same level
                    orders_with_same_price = [order for order in self.active_limit_orders.values() if order.limit_price == price]
                    # if there are no orders with the same price then we generate another order
                    if len(orders_with_same_price) == 0:
                        self.generated_cancel_orders_empty_depth += 1
                        #chech if there are sell limit orders active
                        if len([order for order in self.active_limit_orders.values() if not order.is_buy_order]) == 0:
                            self.drop_counts["cancel_side_empty"] += 1
                            return None
                        order_id = min(self.active_limit_orders.values(), key=lambda x: (abs(x.limit_price - price), abs(x.quantity - size))).order_id
                    else:
                        # we select the order with the quantity near to the quantity generated
                        order_id = min(orders_with_same_price, key=lambda x: abs(x.quantity - size)).order_id
                        self.diff_cancel_order_placed += 1

        elif order_type == 4:
            self.diff_market_order_placed += 1
            _will_exec = True   # Channel A: market orders always execute immediately
            if direction == 1:
                price = self.lob_snapshots[-1][0]
            else:
                price = self.lob_snapshots[-1][2]
            order_id = 0
            # the diffusion gives in output market order and not execution of limit order,
            # so we transform market orders in execution orders of the opposite side as the original message files
            direction = -direction
        self._exec_outcomes.append(1 if _will_exec else 0)   # feed the σ controller
        self.count_diff_placed_orders += 1
        return np.array([time, order_type, order_id, size, price, direction])


    def _update_active_limit_orders(self):
        asks = self.kernel.agents[0].order_books[self.symbol].asks
        bids = self.kernel.agents[0].order_books[self.symbol].bids
        self.active_limit_orders = {}
        for level in asks:
            for order in level:
                self.active_limit_orders[order.order_id] = order
        for level in bids:
            for order in level:
                self.active_limit_orders[order.order_id] = order


    def _z_score_orderbook(self, orderbook):
        if self.fix_lob_pad:
            # H5: training data keeps LOBSTER sentinel prices (+/-9999999999) for missing
            # levels; the sim pads them with 0, which z-scores to a completely different
            # value. Restore the training convention before normalization. Operates on the
            # fresh np.array copy made at the call site, never the stored snapshots
            # (price-reconstruction code relies on `== 0` checks there).
            ask_prices = orderbook[:, 0::4]
            ask_prices[ask_prices == 0] = 9999999999
            bid_prices = orderbook[:, 2::4]
            bid_prices[bid_prices == 0] = -9999999999
        if cst.PRICE_REANCHOR and self.price_anchor:
            # anchor only real quotes: zero-padded missing levels keep their existing (already
            # OOD-extreme) convention, matching how training skips sentinel prices.
            prices = orderbook[:, 0::2]
            _real_quote = (np.abs(prices) > 0) & (np.abs(prices) < 9_000_000_000)
            prices[_real_quote] -= self.price_anchor
        orderbook[:, 0::2] = orderbook[:, 0::2] / 100
        orderbook[:, 0::2] = (orderbook[:, 0::2] - self.normalization_terms["lob"][2]) / self.normalization_terms["lob"][3]
        orderbook[:, 1::2] = (orderbook[:, 1::2] - self.normalization_terms["lob"][0]) / self.normalization_terms["lob"][1]
        return orderbook


    def _preprocess_orders_for_diff_cond(self, orders, lob_snapshots):
        COLUMNS_NAMES = {"orderbook": ["sell1", "vsell1", "buy1", "vbuy1",
                                       "sell2", "vsell2", "buy2", "vbuy2",
                                       "sell3", "vsell3", "buy3", "vbuy3",
                                       "sell4", "vsell4", "buy4", "vbuy4",
                                       "sell5", "vsell5", "buy5", "vbuy5",
                                       "sell6", "vsell6", "buy6", "vbuy6",
                                       "sell7", "vsell7", "buy7", "vbuy7",
                                       "sell8", "vsell8", "buy8", "vbuy8",
                                       "sell9", "vsell9", "buy9", "vbuy9",
                                       "sell10", "vsell10", "buy10", "vbuy10"],
                         "message": ["time", "event_type", "order_id", "size", "price", "direction"]}
        orders_dataframe = pd.DataFrame(orders, columns=COLUMNS_NAMES["message"])
        lob_dataframe = pd.DataFrame(lob_snapshots, columns=COLUMNS_NAMES["orderbook"])

        # we compute the depth of the orders with respect to the orderbook
        orders_dataframe["depth"] = 0
        for j in range(0, orders_dataframe.shape[0]):
            order_price = orders_dataframe["price"].iloc[j]
            direction = orders_dataframe["direction"].iloc[j]
            type = orders_dataframe["event_type"].iloc[j]
            # ALWAYS the pre-event snapshot (lob_dataframe carries one leading row, so index=j is
            # "before orders[j]"). index=j+1 for type==1 was the post-event snapshot — self-referential
            # for a marketable order resting its own remainder (see utils_data.py's matching fix and
            # scripts/check_raw_depth_distribution.py).
            index = j
            if direction == 1:
                bid_side = lob_dataframe.iloc[index, 2::4]
                bid_price = bid_side[0]
                depth = (bid_price - order_price) // 100
                if depth < 0 and not cst.UNCLAMP_DEPTH:   # match training: keep signed depth iff unclamped
                    depth = 0
            else:
                ask_side = lob_dataframe.iloc[index, 0::4]
                ask_price = ask_side[0]
                depth = (order_price - ask_price) // 100
                if depth < 0 and not cst.UNCLAMP_DEPTH:
                    depth = 0
            orders_dataframe.loc[j, "depth"] = depth

        # if order type is 4, then we transform the execution of a sell limit order in a buy market order
        orders_dataframe["direction"] = orders_dataframe["direction"] * orders_dataframe["event_type"].apply(
            lambda x: -1 if x == 4 else 1)

        # drop the order_id column
        orders_dataframe = orders_dataframe.drop(columns=["order_id"])

        # PRICE_REANCHOR: applied after the depth loop above (depth is difference-based) —
        # mirrors preprocess_data's insertion point exactly.
        if cst.PRICE_REANCHOR and self.price_anchor:
            orders_dataframe["price"] = orders_dataframe["price"] - self.price_anchor

        # divide all the price, both of lob and messages, by 100
        orders_dataframe["price"] = orders_dataframe["price"] / 100

        # apply z score to orders
        orders_dataframe, _, _, _, _, _, _, _, _ = normalize_messages(orders_dataframe,
                                                                    mean_size=self.normalization_terms["event"][0],
                                                                    mean_prices=self.normalization_terms["event"][2],
                                                                    std_size=self.normalization_terms["event"][1],
                                                                    std_prices=self.normalization_terms["event"][3],
                                                                    mean_time=self.normalization_terms["event"][4],
                                                                    std_time=self.normalization_terms["event"][5],
                                                                    mean_depth=self.normalization_terms["event"][6],
                                                                    std_depth=self.normalization_terms["event"][7]
                                                                    )

        # Diagnostics: running min/mean/max of the z-scored conditioning channels, so
        # OOD conditioning is visible in the end-of-run report without extra tooling.
        for col in ("time", "size", "price", "depth"):
            if col in orders_dataframe.columns:
                v = orders_dataframe[col].to_numpy(dtype=float)
                if len(v):
                    s = self.cond_stats.setdefault(col, [float("inf"), float("-inf"), 0.0, 0])
                    s[0] = min(s[0], float(v.min()))
                    s[1] = max(s[1], float(v.max()))
                    s[2] += float(v.sum())
                    s[3] += len(v)

        return torch.from_numpy(orders_dataframe.to_numpy()).to(cst.DEVICE, torch.float32)


    def _load_orders_lob(self, symbol, data_dir, date, date_trading_days):
        path = "{}/{}/{}_{}_{}".format(
            data_dir,
            symbol,
            symbol,
            date_trading_days[0],
            date_trading_days[1],
        )
        COLUMNS_NAMES = {"orderbook": ["sell1", "vsell1", "buy1", "vbuy1",
                                       "sell2", "vsell2", "buy2", "vbuy2",
                                       "sell3", "vsell3", "buy3", "vbuy3",
                                       "sell4", "vsell4", "buy4", "vbuy4",
                                       "sell5", "vsell5", "buy5", "vbuy5",
                                       "sell6", "vsell6", "buy6", "vbuy6",
                                       "sell7", "vsell7", "buy7", "vbuy7",
                                       "sell8", "vsell8", "buy8", "vbuy8",
                                       "sell9", "vsell9", "buy9", "vbuy9",
                                       "sell10", "vsell10", "buy10", "vbuy10"],
                         "message": ["time", "event_type", "order_id", "size", "price", "direction"]}
        for i, filename in enumerate(os.listdir(path)):
            f = os.path.join(path, filename)
            filename_splitted = filename.split('_')
            file_date = filename_splitted[1]
            if os.path.isfile(f) and file_date == date:
                if filename_splitted[4] == "message":
                    events = pd.read_csv(f, header=None, names=COLUMNS_NAMES["message"], usecols=range(6))
                elif filename_splitted[4] == "orderbook":
                    lob = pd.read_csv(f, header=None, names=COLUMNS_NAMES["orderbook"])
                else:
                    raise ValueError("File name not recognized")

        events, lob = self._preprocess_events_for_market_replay(events, lob)
        # transform to numpy
        lob = lob.to_numpy()
        events = events.to_numpy()
        return events, lob


    def _preprocess_events_for_market_replay(self, events, lob):

        # drop the rows with event_type = 5, 6, 7
        indexes = events[events["event_type"].isin([5, 6, 7])].index
        events = events.drop(indexes)
        lob = lob.drop(indexes)

        # do the difference of time row per row in messages and subsitute the values with the differences
        first = events["time"].iloc[0]
        events["time"] = events["time"].diff()
        events.iloc[0, events.columns.get_loc("time")] = 0.0

        dataframes = reset_indexes([events, lob])
        events = dataframes[0]
        lob = dataframes[1]
        # get the order ids of the rows with order_type=1
        order_ids = events.loc[events['event_type'] == 1, 'order_id']

        # filter out the rows that have order_type != 1 and have an order id that is not in order_ids
        filtered_df = events.loc[((events['event_type'] != 1) & ~(events['order_id'].isin(order_ids)))].index

        events = events.drop(filtered_df)
        lob = lob.drop(filtered_df)
        dataframes = reset_indexes([events, lob])
        return dataframes[0], dataframes[1]


    def _update_lob_snapshot(self, msg):
        last_lob_snapshot = []
        min_actual_lob_level = min(len(msg.body['asks']), len(msg.body['bids']))
        # we take the first 10 levels of the lob and update the list of lob snapshots
        # to use for the conditioning of the diffusion model
        for i in range(0, 10):
            if i < min_actual_lob_level:
                last_lob_snapshot.append(msg.body['asks'][i][0])
                last_lob_snapshot.append(msg.body['asks'][i][1])
                last_lob_snapshot.append(msg.body['bids'][i][0])
                last_lob_snapshot.append(msg.body['bids'][i][1])
            #we need the else in case the actual lob has less than 10 levels
            else:
                if len(msg.body['asks']) > len(msg.body['bids']) and i < len(msg.body['asks']):
                    last_lob_snapshot.append(msg.body['asks'][i][0])
                    last_lob_snapshot.append(msg.body['asks'][i][1])
                    last_lob_snapshot.append(0)
                    last_lob_snapshot.append(0)
                elif len(msg.body['bids']) > len(msg.body['asks']) and i < len(msg.body['bids']):
                    last_lob_snapshot.append(0)
                    last_lob_snapshot.append(0)
                    last_lob_snapshot.append(msg.body['bids'][i][0])
                    last_lob_snapshot.append(msg.body['bids'][i][1])
                else:
                    for _ in range(4): last_lob_snapshot.append(0)
        self.last_lob_snapshot = last_lob_snapshot
        self.lob_snapshots.append(last_lob_snapshot)
        if len(self.lob_snapshots) > self.seq_len * 4:
            self.lob_snapshots = self.lob_snapshots[-self.seq_len * 4:]
        self.sparse_lob_snapshots.append(to_sparse_representation(last_lob_snapshot, 100))
        if len(self.sparse_lob_snapshots) > self.seq_len * 4:
            self.sparse_lob_snapshots = self.sparse_lob_snapshots[-self.seq_len * 4:]
        
        
    def _preprocess_market_features_for_cgan(self, lob_snapshots):
        lob_snapshots = np.array(lob_snapshots)
        COLUMNS_NAMES = {"orderbook": ["sell1", "vsell1", "buy1", "vbuy1",
                                       "sell2", "vsell2", "buy2", "vbuy2",
                                       "sell3", "vsell3", "buy3", "vbuy3",
                                       "sell4", "vsell4", "buy4", "vbuy4",
                                       "sell5", "vsell5", "buy5", "vbuy5",
                                       "sell6", "vsell6", "buy6", "vbuy6",
                                       "sell7", "vsell7", "buy7", "vbuy7",
                                       "sell8", "vsell8", "buy8", "vbuy8",
                                       "sell9", "vsell9", "buy9", "vbuy9",
                                       "sell10", "vsell10", "buy10", "vbuy10"],
                        }
        lob_dataframe = pd.DataFrame(lob_snapshots, columns=COLUMNS_NAMES["orderbook"])
        orders = np.array(self.placed_orders[-self.seq_len*2 +1:])
        orders_dataframe = pd.DataFrame(orders, columns=["time", "type", "order_id", "quantity", "price", "direction"])
        dataframes = [[orders_dataframe, lob_dataframe]]
        mean_spread = self.normalization_terms["lob"][0]
        std_spread = self.normalization_terms["lob"][1]
        mean_return = self.normalization_terms["lob"][2]
        std_return = self.normalization_terms["lob"][3]
        mean_vol_imb = self.normalization_terms["lob"][4]
        std_vol_imb = self.normalization_terms["lob"][5]
        mean_abs_vol = self.normalization_terms["lob"][6]
        std_abs_vol = self.normalization_terms["lob"][7]
        for i in range(len(dataframes)):
            lob_sizes = dataframes[i][1].iloc[:, 1::2]
            lob_prices = dataframes[i][1].iloc[:, 0::2]
            dataframes[i][1]["volume_imbalance_1"] = lob_sizes.iloc[:, 1] / (lob_sizes.iloc[:, 1] + lob_sizes.iloc[:, 0])
            dataframes[i][1]["volume_imbalance_5"] = (lob_sizes.iloc[:, 1] + lob_sizes.iloc[:, 3] + lob_sizes.iloc[:, 5] + lob_sizes.iloc[:, 7] + lob_sizes.iloc[:, 9]) / (lob_sizes.iloc[:, :10].sum(axis=1))
            dataframes[i][1]["absolute_volume_1"] = lob_sizes.iloc[:, 1] + lob_sizes.iloc[:, 0]
            dataframes[i][1]["absolute_volume_5"] = lob_sizes.iloc[:, :10].sum(axis=1)
            dataframes[i][1]["spread"] = lob_prices.iloc[:, 0] - lob_prices.iloc[:, 1]

        for i in range(len(dataframes)):
            order_sign_imbalance_256 = pd.Series(0, index=dataframes[i][1].index)
            order_sign_imbalance_128 = pd.Series(0, index=dataframes[i][1].index)
            returns_50 = pd.Series(0, index=dataframes[i][1].index)
            returns_1 = pd.Series(0, index=dataframes[i][1].index)
            lob_prices = dataframes[i][1].iloc[:, 0::2]
            mid_prices = (lob_prices.iloc[:, 0] + lob_prices.iloc[:, 1]) / 2
            for j in range(len(dataframes[i][1])-256):
                order_sign_imbalance_256.iloc[j] = dataframes[i][0]["direction"].iloc[j:j+256].sum()
                order_sign_imbalance_128.iloc[j] = dataframes[i][0]["direction"].iloc[j+128:j+256].sum()
                returns_1 = returns_1.astype(float)
                returns_1.iloc[j] = mid_prices[j+255] / mid_prices[j+254] - 1
                returns_50 = returns_50.astype(float)
                returns_50.iloc[j] = mid_prices[j+255] / mid_prices[j+205] - 1
            dataframes[i][1] = dataframes[i][1].iloc[255:]
            dataframes[i][1].loc[:, "order_sign_imbalance_256"] = order_sign_imbalance_256.iloc[:-255] / 256
            dataframes[i][1].loc[:, "order_sign_imbalance_128"] = order_sign_imbalance_128.iloc[:-255] / 128
            dataframes[i][1].loc[:, "returns_1"] = returns_1.iloc[:-255]
            dataframes[i][1].loc[:, "returns_50"] = returns_50.iloc[:-255]
            dataframes[i][1] = dataframes[i][1][["volume_imbalance_1", "volume_imbalance_5", "absolute_volume_1", "absolute_volume_5", "spread", "order_sign_imbalance_256", "order_sign_imbalance_128", "returns_1", "returns_50"]]
        
        for i in range(len(dataframes)):
            dataframes[i][1] = dataframes[i][1].reset_index(drop=True)
        
        for i in range(len(dataframes)):
            #transform nan values in 0
            dataframes[i][1] = dataframes[i][1].fillna(0)

        market_features = dataframes[0][1]
        market_features["returns_1"] = (market_features["returns_1"] - mean_return) / std_return
        market_features["returns_50"] = (market_features["returns_50"] - mean_return) / std_return
        market_features["volume_imbalance_1"] = (market_features["volume_imbalance_1"] - mean_vol_imb) / std_vol_imb
        market_features["volume_imbalance_5"] = (market_features["volume_imbalance_5"] - mean_vol_imb) / std_vol_imb
        market_features["absolute_volume_1"] = (market_features["absolute_volume_1"] - mean_abs_vol) / std_abs_vol
        market_features["absolute_volume_5"] = (market_features["absolute_volume_5"] - mean_abs_vol) / std_abs_vol
        market_features["spread"] = (market_features["spread"] - mean_spread) / std_spread
        market_features = market_features.to_numpy()
        market_features = torch.from_numpy(market_features).to(cst.DEVICE, torch.float32)
        market_features = market_features.unsqueeze(0)
        return market_features












