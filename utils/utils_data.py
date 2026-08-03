import json
import pandas as pd
import numpy as np
import os

import torch
import pandas
import constants as cst

def z_score_market_features(data, mean_spread=None, mean_returns=None, mean_vol_imb=None, mean_abs_vol=None, std_spread=None, std_returns=None, std_vol_imb=None, std_abs_vol=None):
    data = data.reset_index(drop=True)
    if (mean_spread is None) or (std_spread is None):
        mean_spread = data["spread"].mean()
        std_spread = data["spread"].std()
    
    if (mean_returns is None) or (std_returns is None):
        #concatenates returns_1 and returns_5
        mean_returns = pd.concat([data["returns_1"], data["returns_50"]]).mean()
        std_returns = pd.concat([data["returns_1"], data["returns_50"]]).std()
        
    if (mean_vol_imb is None) or (std_vol_imb is None):
        mean_vol_imb = pd.concat([data["volume_imbalance_1"], data["volume_imbalance_5"]]).mean()
        std_vol_imb = pd.concat([data["volume_imbalance_1"], data["volume_imbalance_5"]]).std()
        
    if (mean_abs_vol is None) or (std_abs_vol is None):
        mean_abs_vol = pd.concat([data["absolute_volume_1"], data["absolute_volume_5"]]).mean()
        std_abs_vol = pd.concat([data["absolute_volume_1"], data["absolute_volume_5"]]).std()
    
    data["spread"] = (data["spread"] - mean_spread) / std_spread
    data["returns_1"] = (data["returns_1"] - mean_returns) / std_returns
    data["returns_50"] = (data["returns_50"] - mean_returns) / std_returns
    data["volume_imbalance_1"] = (data["volume_imbalance_1"] - mean_vol_imb) / std_vol_imb
    data["volume_imbalance_5"] = (data["volume_imbalance_5"] - mean_vol_imb) / std_vol_imb
    data["absolute_volume_1"] = (data["absolute_volume_1"] - mean_abs_vol) / std_abs_vol
    data["absolute_volume_5"] = (data["absolute_volume_5"] - mean_abs_vol) / std_abs_vol
    print()
    print("mean spread ", mean_spread)
    print("std spread ", std_spread)
    print("mean returns ", mean_returns)
    print("std returns ", std_returns)
    print("mean vol imb ", mean_vol_imb)
    print("std vol imb ", std_vol_imb)
    print("mean abs vol ", mean_abs_vol)
    print("std abs vol ", std_abs_vol)
    print(data[:10])
    print()
    return data, mean_spread, mean_returns, mean_vol_imb, mean_abs_vol, std_spread, std_returns, std_vol_imb, std_abs_vol



def normalize_order_cgan(data, mean_size=None, mean_depth=None, mean_cancel_depth=None, mean_size_100=None, std_size=None, std_depth=None, std_cancel_depth=None, std_size_100=None):
    data = data.reset_index(drop=True)
    if (mean_size is None) or (std_size is None):
        mean_size = data["size"].mean()
        std_size = data["size"].std()
    
    if (mean_depth is None) or (std_depth is None):
        mean_depth = data["depth"].mean()
        std_depth = data["depth"].std()
        
    if (mean_cancel_depth is None) or (std_cancel_depth is None):
        mean_cancel_depth = data["cancel_depth"].mean()
        std_cancel_depth = data["cancel_depth"].std()
        
    if (mean_size_100 is None) or (std_size_100 is None):
        mean_size_100 = data["quantity_100"].mean()
        std_size_100 = data["quantity_100"].std()
        
    data["size"] = (data["size"] - mean_size) / std_size
    data["depth"] = (data["depth"] - mean_depth) / std_depth
    data["cancel_depth"] = (data["cancel_depth"] - mean_cancel_depth) / std_cancel_depth
    data["quantity_100"] = (data["quantity_100"] - mean_size_100) / std_size_100
    
    data["event_type"] = data["event_type"]-1.0
    data["event_type"] = data["event_type"].replace(2, 1)
    data["event_type"] = data["event_type"].replace(3, 2)
    data["event_type"] = data["event_type"]-1.0
    # order_type = -1 -> limit order
    # order_type = 0 -> cancel order
    # order_type = 1 -> market order
    print("mean size order cgan", mean_size)
    print("std size order cgan", std_size)
    print("mean depth order cgan", mean_depth)
    print("std depth order cgan", std_depth)
    print("mean cancel depth order cgan", mean_cancel_depth)
    print("std cancel depth order cgan", std_cancel_depth)
    print("mean size 100 order cgan", mean_size_100)
    print("std size 100 order cgan", std_size_100)
    print(data[:5])
    
    return data, mean_size, mean_depth, mean_cancel_depth, mean_size_100, std_size, std_depth, std_cancel_depth, std_size_100


def z_score_orderbook(data, mean_size=None, mean_prices=None, std_size=None, std_prices=None):
    """ DONE: remember to use the mean/std of the training set, to z-normalize the test set. """
    if (mean_size is None) or (std_size is None):
        mean_size = data.iloc[:, 1::2].stack().mean()
        std_size = data.iloc[:, 1::2].stack().std()

    #do the same thing for prices
    if (mean_prices is None) or (std_prices is None):
        mean_prices = data.iloc[:, 0::2].stack().mean()
        std_prices = data.iloc[:, 0::2].stack().std()
    if std_size == 0 or pd.isna(std_size):
        std_size = 1.0
    if std_prices == 0 or pd.isna(std_prices):
        std_prices = 1.0
    # apply the z score to the original data using .loc with explicit float cast
    price_cols = data.columns[0::2]
    size_cols = data.columns[1::2]

    #apply the z score to the original data
    for col in size_cols:
        data[col] = data[col].astype("float64")
        data[col] = (data[col] - mean_size) / std_size

    for col in price_cols:
        data[col] = data[col].astype("float64")
        data[col] = (data[col] - mean_prices) / std_prices

    # check if there are null values, then raise value error
    if data.isnull().values.any():
        raise ValueError("data contains null value")

    return data, mean_size, mean_prices, std_size,  std_prices


def normalize_messages(data, mean_size=None, mean_prices=None, std_size=None,  std_prices=None, mean_time=None, std_time=None, mean_depth=None, std_depth=None):

    #apply z score to prices and size column
    if (mean_size is None) or (std_size is None):
        mean_size = data["size"].mean()
        std_size = data["size"].std()

    if (mean_prices is None) or (std_prices is None):
        mean_prices = data["price"].mean()
        std_prices = data["price"].std()

    if (mean_time is None) or (std_time is None):
        mean_time = data["time"].mean()
        std_time = data["time"].std()

    if (mean_depth is None) or (std_depth is None):
        mean_depth = data["depth"].mean()
        std_depth = data["depth"].std()

    #apply the z score to the original data
    data["time"] = (data["time"] - mean_time) / std_time
    data["size"] = (data["size"] - mean_size) / std_size
    data["price"] = (data["price"] - mean_prices) / std_prices
    data["depth"] = (data["depth"] - mean_depth) / std_depth

    # check if there are null values, then raise value error
    if data.isnull().values.any():
        raise ValueError("data contains null value")

    data["event_type"] = data["event_type"]-1.0
    data["event_type"] = data["event_type"].replace(2, 1)
    data["event_type"] = data["event_type"].replace(3, 2)
    # order_type = 0 -> limit order
    # order_type = 1 -> cancel order
    # order_type = 2 -> market order

    return data, mean_size, mean_prices, std_size,  std_prices, mean_time, std_time, mean_depth, std_depth


def z_score_orderbook_for_cond(orderbook, normalization_terms, price_anchor=0.0,
                                fix_lob_pad=False, cond_clip=0.0):
    """Z-score a raw LOB snapshot array for model conditioning.

    Extracted from WorldAgent._z_score_orderbook (ABIDES/agent/WorldAgent.py) so
    the exact same conditioning logic is shared between live simulation and the
    RL cold-start module (rl_execution/coldstart.py) -- must not diverge, see
    the cond_z diagnostic drift warning in the RL execution project spec.

    orderbook: np.ndarray, one row per LOB snapshot, columns
        [sell1, vsell1, buy1, vbuy1, sell2, vsell2, ...] in raw LOBSTER units.
        Mutated in place (matches the original method's behavior).
    normalization_terms: dict with "lob" -> (mean_size, std_size, mean_price, std_price).
    price_anchor: day's opening mid, in raw price units; applied only if nonzero
        (callers gate this on cst.PRICE_REANCHOR, same as the original method).
    fix_lob_pad: H5 sentinel-padding convention, see original method.
    cond_clip: clip z-scored sizes to [-cond_clip, cond_clip]; 0 = off.

    Returns (orderbook, clipped_count) -- clipped_count is how many size entries
    were clipped, for callers that want to accumulate a running diagnostic
    (WorldAgent accumulates this into self.cond_clipped_count).
    """
    if fix_lob_pad:
        ask_prices = orderbook[:, 0::4]
        ask_prices[ask_prices == 0] = 9999999999
        bid_prices = orderbook[:, 2::4]
        bid_prices[bid_prices == 0] = -9999999999
    if price_anchor:
        prices = orderbook[:, 0::2]
        _real_quote = (np.abs(prices) > 0) & (np.abs(prices) < 9_000_000_000)
        prices[_real_quote] -= price_anchor
    orderbook[:, 0::2] = orderbook[:, 0::2] / 100
    orderbook[:, 0::2] = (orderbook[:, 0::2] - normalization_terms["lob"][2]) / normalization_terms["lob"][3]
    orderbook[:, 1::2] = (orderbook[:, 1::2] - normalization_terms["lob"][0]) / normalization_terms["lob"][1]
    clipped_count = 0
    if cond_clip > 0.0:
        sizes = orderbook[:, 1::2]
        clipped_count = int(np.count_nonzero(np.abs(sizes) > cond_clip))
        np.clip(sizes, -cond_clip, cond_clip, out=sizes)
    return orderbook, clipped_count


def preprocess_orders_for_diff_cond(orders, lob_snapshots, normalization_terms,
                                     price_anchor=0.0, cond_stats=None):
    """Build the z-scored order-history conditioning tensor for the diffusion model.

    Extracted from WorldAgent._preprocess_orders_for_diff_cond (see
    z_score_orderbook_for_cond's docstring above for why this is shared rather
    than reimplemented independently in the RL cold-start module).

    orders: np.ndarray [n, 6], columns (time, event_type, order_id, size, price, direction).
    lob_snapshots: np.ndarray [n+1, 40] -- one leading pre-event row, so row j is
        always the LOB state immediately BEFORE orders[j] (see the depth-computation
        comment below for why this indexing matters).
    normalization_terms: dict with "event" -> (mean_size, std_size, mean_price,
        std_price, mean_time, std_time, mean_depth, std_depth).
    price_anchor: day's opening mid, in raw price units; applied only if nonzero.
    cond_stats: optional dict, mutated in place, accumulating running
        [min, max, sum, count] per z-scored channel (time/size/price/depth) --
        the cond_z diagnostic. Pass a fresh {} to get just this call's stats
        (e.g. at cold-start seed time), or a persistent dict to accumulate
        across an episode/run (as WorldAgent does with self.cond_stats).

    Returns a torch.FloatTensor on cst.DEVICE, columns (time, event_type, size,
    price, direction, depth) -- order_id is dropped.
    """
    columns = ["time", "event_type", "order_id", "size", "price", "direction"]
    orderbook_columns = [c for i in range(1, 11) for c in
                         (f"sell{i}", f"vsell{i}", f"buy{i}", f"vbuy{i}")]
    orders_dataframe = pd.DataFrame(orders, columns=columns)
    lob_dataframe = pd.DataFrame(lob_snapshots, columns=orderbook_columns)

    # compute the depth of each order with respect to the orderbook
    orders_dataframe["depth"] = 0
    for j in range(0, orders_dataframe.shape[0]):
        order_price = orders_dataframe["price"].iloc[j]
        direction = orders_dataframe["direction"].iloc[j]
        # ALWAYS the pre-event snapshot (lob_dataframe carries one leading row, so
        # index=j is "before orders[j]").
        index = j
        if direction == 1:
            bid_side = lob_dataframe.iloc[index, 2::4]
            # .iloc[0]: positional, not label-based. Plain [0] on a Series is deprecated
            # (pandas FutureWarning) and in a future pandas would silently switch to
            # label lookup -- which here would raise, since this Series is indexed by
            # column NAME ("buy1", ...), not by integer.
            bid_price = bid_side.iloc[0]
            depth = (bid_price - order_price) // 100
            if depth < 0 and not cst.UNCLAMP_DEPTH:  # match training: keep signed depth iff unclamped
                depth = 0
        else:
            ask_side = lob_dataframe.iloc[index, 0::4]
            ask_price = ask_side.iloc[0]
            depth = (order_price - ask_price) // 100
            if depth < 0 and not cst.UNCLAMP_DEPTH:
                depth = 0
        orders_dataframe.loc[j, "depth"] = depth

    # if order type is 4, then we transform the execution of a sell limit order into a buy market order
    orders_dataframe["direction"] = orders_dataframe["direction"] * orders_dataframe["event_type"].apply(
        lambda x: -1 if x == 4 else 1)

    orders_dataframe = orders_dataframe.drop(columns=["order_id"])

    # PRICE_REANCHOR: applied after the depth loop above (depth is difference-based) --
    # mirrors preprocess_data's insertion point exactly.
    if price_anchor:
        orders_dataframe["price"] = orders_dataframe["price"] - price_anchor

    orders_dataframe["price"] = orders_dataframe["price"] / 100

    orders_dataframe, _, _, _, _, _, _, _, _ = normalize_messages(
        orders_dataframe,
        mean_size=normalization_terms["event"][0],
        mean_prices=normalization_terms["event"][2],
        std_size=normalization_terms["event"][1],
        std_prices=normalization_terms["event"][3],
        mean_time=normalization_terms["event"][4],
        std_time=normalization_terms["event"][5],
        mean_depth=normalization_terms["event"][6],
        std_depth=normalization_terms["event"][7],
    )

    if cond_stats is not None:
        for col in ("time", "size", "price", "depth"):
            if col in orders_dataframe.columns:
                v = orders_dataframe[col].to_numpy(dtype=float)
                if len(v):
                    s = cond_stats.setdefault(col, [float("inf"), float("-inf"), 0.0, 0])
                    s[0] = min(s[0], float(v.min()))
                    s[1] = max(s[1], float(v.max()))
                    s[2] += float(v.sum())
                    s[3] += len(v)

    return torch.from_numpy(orders_dataframe.to_numpy()).to(cst.DEVICE, torch.float32)


def load_compute_normalization_terms(stock_name, data_dir, model, n_lob_levels):
    """Return normalization stats for WorldAgent/simulation.

    Tries the cached JSON written by preprocessing first (fast path).
    Falls back to recomputing from raw LOBSTER CSVs if the JSON is absent.
    """
    if model == cst.Models.TRADES:
        stats_path = os.path.join(data_dir, stock_name, "normalization_stats.json")
        if os.path.exists(stats_path):
            with open(stats_path) as f:
                s = json.load(f)
            lob = s["lob"]
            evt = s["event"]
            print(f"[utils_data] Loaded normalization stats from {stats_path}")
            return {
                "lob":   (lob["mean_size"], lob["std_size"], lob["mean_price"], lob["std_price"]),
                "event": (evt["mean_size"], evt["std_size"], evt["mean_price"], evt["std_price"],
                          evt["mean_time"], evt["std_time"], evt["mean_depth"], evt["std_depth"]),
            }
        print(f"[utils_data] normalization_stats.json not found for {stock_name}; recomputing from raw data")

    # original slow path: read raw CSVs and recompute
    return _load_compute_normalization_terms_slow(stock_name, data_dir, model, n_lob_levels)


def _load_compute_normalization_terms_slow(stock_name, data_dir, model, n_lob_levels):
    path = "{}/{}/{}_{}_{}".format(
            data_dir,
            stock_name,
            stock_name,
            cst.DATE_TRADING_DAYS[0],
            cst.DATE_TRADING_DAYS[-1]
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
    
    num_trading_days = len(os.listdir(path))//2
    split_rates = cst.SPLIT_RATES
    train = int(round(num_trading_days * split_rates[0]))
    val = int(round(num_trading_days * split_rates[1])) + train
    test = int(round(num_trading_days * split_rates[2])) + val
    split_days = [train, val, test]
    split_days = [i * 2 for i in split_days]
    for i, filename in enumerate(sorted(os.listdir(path))):
        f = os.path.join(path, filename)
        if os.path.isfile(f):
            # then we create the df for the training set
            if i < split_days[0]:
                if (i % 2) == 0:
                    if i == 0:
                        train_messages = pd.read_csv(f, names=COLUMNS_NAMES["message"], usecols=range(6))
                    else:
                        train_message = pd.read_csv(f, names=COLUMNS_NAMES["message"], usecols=range(6))

                else:
                    if i == 1:
                        train_orderbooks = pd.read_csv(f, names=COLUMNS_NAMES["orderbook"])
                        train_orderbooks, train_messages = preprocess_data([train_messages, train_orderbooks], n_lob_levels, model)
                        if (len(train_orderbooks) != len(train_messages)):
                            raise ValueError("train_orderbook length is different than train_messages")
                    else:
                        train_orderbook = pd.read_csv(f, names=COLUMNS_NAMES["orderbook"])
                        train_orderbook, train_message = preprocess_data([train_message, train_orderbook], n_lob_levels, model)
                        train_messages = pd.concat([train_messages, train_message], axis=0)
                        train_orderbooks = pd.concat([train_orderbooks, train_orderbook], axis=0)
    if model == cst.Models.TRADES:
        train_orderbooks = train_orderbooks.astype(float)
        train_orderbooks.loc[:, ::2] /= 100
        train_messages["price"] /= 100
        _, lob_mean_size, lob_mean_prices, lob_std_size, lob_std_prices = z_score_orderbook(train_orderbooks)
        _, mean_size, mean_prices, std_size,  std_prices, mean_time, std_time, mean_depth, std_depth = normalize_messages(train_messages)
        normalization_terms = {
            "lob": (lob_mean_size, lob_std_size, lob_mean_prices, lob_std_prices),
            "event": (mean_size, std_size, mean_prices, std_prices, mean_time, std_time, mean_depth, std_depth)
        }
        return normalization_terms
    elif model == cst.Models.CGAN:
        _, mean_spread, mean_returns, mean_vol_imb, mean_abs_vol, std_spread, std_returns, std_vol_imb, std_abs_vol = z_score_market_features(train_orderbooks)
        _, mean_size, mean_depth, mean_cancel_depth, mean_size_100, std_size, std_depth, std_cancel_depth, std_size_100 = normalize_order_cgan(train_messages)
        normalization_terms = {
            "lob": (mean_spread, std_spread, mean_returns, std_returns, mean_vol_imb, std_vol_imb, mean_abs_vol, std_abs_vol, mean_cancel_depth, std_cancel_depth, mean_size_100, std_size_100, mean_depth, std_depth, mean_size, std_size),
        }
        return normalization_terms

def reset_indexes(dataframes):
    # reset the indexes of the messages and orderbooks
    dataframes[0] = dataframes[0].reset_index(drop=True)
    dataframes[1] = dataframes[1].reset_index(drop=True)
    return dataframes


_LOB_SENTINEL = 9_000_000_000  # |price| above this = LOBSTER missing-level sentinel


def compute_price_anchor(orderbook):
    """Day anchor for PRICE_REANCHOR: the first row's valid best-bid/ask mid, in RAW price units.
    Shared by training preprocessing AND WorldAgent so the convention cannot diverge.
    Accepts a DataFrame or ndarray with the standard 40-col layout (ask1 price at col 0,
    bid1 price at col 2)."""
    arr = np.asarray(orderbook, dtype=float)
    for j in range(len(arr)):
        a1, b1 = arr[j, 0], arr[j, 2]
        if 0 < a1 < _LOB_SENTINEL and 0 < b1 < _LOB_SENTINEL:
            return round((a1 + b1) / 2.0)
    raise ValueError("compute_price_anchor: no row with a valid two-sided book")


def preprocess_data(dataframes, n_lob_levels, chosen_model):
    print("n_lob_levels =", n_lob_levels, "-> LOB cols =", n_lob_levels * cst.LEN_LEVEL)
    print("orderbook width in:", dataframes[1].shape[1])
    dataframes = reset_indexes(dataframes)

    # take only the first n_lob_levels levels of the orderbook and drop the others
    dataframes[1] = dataframes[1].iloc[:, :n_lob_levels * cst.LEN_LEVEL]

    # take the indexes of the dataframes that are of type 
    # 2 (partial deletion), 5 (execution of a hidden limit order), 
    # 6 (cross trade), 7 (trading halt) and drop it
    indexes_to_drop = dataframes[0][dataframes[0]["event_type"].isin([2, 5, 6, 7])].index
    dataframes[0] = dataframes[0].drop(indexes_to_drop)
    dataframes[1] = dataframes[1].drop(indexes_to_drop)

    dataframes = reset_indexes(dataframes)

    # drop index column in messages
    dataframes[0] = dataframes[0].drop(columns=["order_id"])

    # do the difference of time row per row in messages and subsitute the values with the differences
    # Store the initial value of the "time" column
    first_time = dataframes[0]["time"].values[0]
    # Calculate the difference using diff
    dataframes[0]["time"] = dataframes[0]["time"].diff()
    # Set the first value directly
    dataframes[0].iat[0, dataframes[0].columns.get_loc("time")] = 0.0
        
    # add depth column to messages
    dataframes[0]["depth"] = 0

    # we compute the depth of the orders with respect to the orderbook
    # Extract necessary columns
    prices = dataframes[0]["price"].values
    directions = dataframes[0]["direction"].values
    event_types = dataframes[0]["event_type"].values
    bid_sides = dataframes[1].iloc[:, 2::4].values
    ask_sides = dataframes[1].iloc[:, 0::4].values
    
    # Initialize depth array
    depths = np.zeros(dataframes[0].shape[0], dtype=int)

    # Compute the depth of the orders with respect to the orderbook
    for j in range(1, len(prices)):
        order_price = prices[j]
        direction = directions[j]
        event_type = event_types[j]
        
        # ALWAYS the pre-event orderbook snapshot (LOBSTER's orderbook row j = state AFTER message
        # row j, so j-1 is "before"). Using index=j for event_type==1 (post-event) was
        # self-referential: a marketable order that rests its own remainder becomes the new best
        # bid/ask it's being compared against, washing depth to 0 instead of negative — which is why
        # UNCLAMP_DEPTH found zero negatives in real data regardless of the clamp (confirmed via
        # scripts/check_raw_depth_distribution.py: 0.00% negative before this fix, 0.91% of LIMIT
        # events after).
        index = j - 1

        if direction == 1:
            bid_price = bid_sides[index, 0]
            depth = (bid_price - order_price) // 100
        else:
            ask_price = ask_sides[index, 0]
            depth = (order_price - ask_price) // 100
        
        depths[j] = depth if cst.UNCLAMP_DEPTH else max(depth, 0)  # UNCLAMP_DEPTH keeps marketable (depth<0)
    
    # Assign the computed depths back to the DataFrame
    dataframes[0]["depth"] = depths
        
    # we eliminate the first row of every dataframe because we can't deduce the depth
    dataframes[0] = dataframes[0].iloc[1:, :]
    dataframes[1] = dataframes[1].iloc[1:, :]

    dataframes = reset_indexes(dataframes)

    # PRICE_REANCHOR: subtract the day's opening mid from every price (messages + LOB), in raw
    # units, AFTER depth computation (depth is difference-based, so provably unaffected) and
    # BEFORE the /100 + z-scoring downstream — both the builder and the slow stats path inherit
    # anchored prices automatically, so normalization stats become intraday-deviation stats
    # (mean_price ~0 instead of ~3620). Sentinel prices (|.|>9e9, missing levels) are skipped so
    # they stay bit-identical to the unanchored convention. TRADES-focused: the flag should stay
    # off for CGAN (its returns_1/returns_50 use pct_change, whose denominator anchoring shifts).
    if cst.PRICE_REANCHOR:
        anchor = compute_price_anchor(dataframes[1])
        dataframes[0]["price"] = dataframes[0]["price"] - anchor
        lob_price_cols = dataframes[1].columns[0::2]
        for col in lob_price_cols:
            v = dataframes[1][col].astype("float64")
            mask = v.abs() < _LOB_SENTINEL
            dataframes[1][col] = v.where(~mask, v - anchor)
        print(f"[preprocess_data] PRICE_REANCHOR on: day anchor = {anchor} (raw units)")

    if chosen_model == cst.Models.CGAN:
        # Initialize new columns
        dataframes[0]["cancel_depth"] = 0
        dataframes[0]["quantity_100"] = dataframes[0]["size"].apply(lambda x: x // 100 if x % 100 == 0 else 0)
        dataframes[0]["quantity_type"] = dataframes[0]["size"].apply(lambda x: -1 if x % 100 == 0 else 1)

        # Calculate cancel_depth using vectorization
        cancel_mask = dataframes[0]["event_type"] == 3
        shifted_prices = dataframes[1].shift(1).bfill()
        price_levels = shifted_prices.iloc[:, ::2].apply(lambda row: dict(zip(row, range(0, len(row)*2, 2))), axis=1)
        dataframes[0].loc[cancel_mask, "cancel_depth"] = dataframes[0].loc[cancel_mask].apply(
            lambda row: price_levels[row.name].get(row["price"], np.nan) // 2, axis=1
        )

        # Drop unnecessary columns
        dataframes[0] = dataframes[0].drop(columns=["price", "time"])

        # Shift and fill NaN values
        dataframes[1] = dataframes[1].shift(1).fillna(0)

        # Calculate volume imbalances, absolute volumes, and spread
        lob_sizes = dataframes[1].iloc[:, 1::2]  # Even columns (size)
        lob_prices = dataframes[1].iloc[:, 0::2]  # Odd columns (price)

        # Volume imbalance for level 1
        dataframes[1]["volume_imbalance_1"] = lob_sizes.iloc[:, 1] / (lob_sizes.iloc[:, 1] + lob_sizes.iloc[:, 0])

        # Volume imbalance for levels 1-5
        best_5_asks = lob_sizes.iloc[:, 1:10:2]  # Columns 1,3,5,7,9
        best_5_bids = lob_sizes.iloc[:, 0:10:2]  # Columns 0,2,4,6,8
        dataframes[1]["volume_imbalance_5"] = best_5_asks.sum(axis=1) / (best_5_asks.sum(axis=1) + best_5_bids.sum(axis=1))

        # Absolute volumes
        dataframes[1]["absolute_volume_1"] = lob_sizes.iloc[:, 1] + lob_sizes.iloc[:, 0]
        dataframes[1]["absolute_volume_5"] = (lob_sizes.iloc[:, :10]).sum(axis=1)

        # Spread
        dataframes[1]["spread"] = lob_prices.iloc[:, 0] - lob_prices.iloc[:, 1]

        # Calculate mid prices
        mid_prices = (lob_prices.iloc[:, 0] + lob_prices.iloc[:, 1]) / 2

        # Calculate order sign imbalances and returns using rolling sums
        dataframes[0]["cumulative_direction"] = dataframes[0]["direction"].cumsum()
        dataframes[1]["order_sign_imbalance_256"] = dataframes[0]["cumulative_direction"] - dataframes[0]["cumulative_direction"].shift(256, fill_value=0)
        dataframes[1]["order_sign_imbalance_128"] = dataframes[0]["cumulative_direction"].shift(128, fill_value=0) - dataframes[0]["cumulative_direction"].shift(256, fill_value=0)

        # Returns
        dataframes[1]["returns_1"] = mid_prices.pct_change(periods=1).shift(-1)
        dataframes[1]["returns_50"] = mid_prices.pct_change(periods=50).shift(-50)

        # Trim the first 255 rows
        dataframes[0] = dataframes[0].iloc[256:].reset_index(drop=True)
        dataframes[1] = dataframes[1].iloc[256:].reset_index(drop=True)

        # Select required columns
        dataframes[1] = dataframes[1][[
            "volume_imbalance_1", "volume_imbalance_5",
            "absolute_volume_1", "absolute_volume_5",
            "spread", "order_sign_imbalance_256",
            "order_sign_imbalance_128", "returns_1", "returns_50"
        ]]

        # Fill NaN values
        dataframes[0] = dataframes[0].fillna(0)
        dataframes[1] = dataframes[1].fillna(0)
    
    # we transform the execution of a sell limit order in a buy market order and viceversa
    dataframes[0]["direction"] = dataframes[0]["direction"] * dataframes[0]["event_type"].apply(
        lambda x: -1 if x == 4 else 1)
    print("OUT -> lob:", dataframes[1].shape, "orders:", dataframes[0].shape)
    return dataframes[1], dataframes[0]


def unnormalize(x, mean, std):
    return x * std + mean


def one_hot_encoding_type(data):
    encoded_data = torch.zeros(data.shape[0], data.shape[1] + 2, dtype=torch.float32)
    encoded_data[:, 0] = data[:, 0]
    # encoding order type
    one_hot_order_type = torch.nn.functional.one_hot((data[:, 1]).to(torch.int64), num_classes=3).to(
        torch.float32)
    encoded_data[:, 1:4] = one_hot_order_type
    encoded_data[:, 4:] = data[:, 2:]
    return encoded_data


def tanh_encoding_type(data):
    data[:, 1] = torch.where(data[:, 1] == 1.0, 2.0, torch.where(data[:, 1] == 2.0, 1.0, data[:, 1]))
    data[:, 1] = data[:, 1] - 1
    return data


def to_sparse_representation(lob, n_levels):
    if not isinstance(lob, np.ndarray):
        lob = np.array(lob)
    sparse_lob = np.zeros(n_levels * 2)
    for j in range(lob.shape[0] // 2):
        if j % 2 == 0:
            ask_price = lob[0]
            current_ask_price = lob[j*2]
            depth = (current_ask_price - ask_price) // 100
            if depth < n_levels and int(lob[j*2]) != 0:
                sparse_lob[2*int(depth)] = lob[j*2+1]
        else:
            bid_price = lob[2]
            current_bid_price = lob[j*2]
            depth = (bid_price - current_bid_price) // 100
            if depth < n_levels and int(lob[j*2]) != 0:
                sparse_lob[2*int(depth)+1] = lob[j*2+1]
    return sparse_lob
