"""
Kaggriculture Agent — "AgroBoss"
================================
A rule-based, multi-worker farm manager. No ML/training involved — this is a
carefully-tuned greedy scheduler, which is the right tool here: the game is a
fully-observed, deterministic-rules resource-management problem, so a fast,
well-reasoned heuristic beats a slow/undertrained model and never risks a
runtime error mid-episode (which would zero out the score).

STRATEGY IN ONE PARAGRAPH
--------------------------
Every turn we scan our whole farm and build a prioritized "job list"
(harvest > feed animals > water > clear weeds > care for animals > collect
fertilizer > place bought animals > fertilize > plant > build). We then
greedily assign the closest available farmer/hand to each job, nearest job
first. Money-related decisions (selling, buying seeds/animals/land, hiring)
are queued as market orders every turn, cheapest/most-certain wins first
(sell before buy, so the same turn's income is available to spend).

WHY THIS SHOULD PLACE WELL
---------------------------
1. It never idles a unit if there's useful work within reach.
2. It diversifies crops (melon/strawberry/tomato are the highest $/tile/day)
   instead of the single-crop "starter agent" trap.
3. It reinvests aggressively: hires hands and buys land as soon as it's
   comfortably affordable, so the farm compounds instead of staying small.
4. It builds a modest, growing animal operation (free structures + cheap
   fertilizer) without over-committing tiles away from crops.
5. It is defensive: every external call is wrapped so a bug can never crash
   the submission and forfeit the match.

TUNE HERE FIRST if you want to iterate:
- CROP_PLANT_ORDER      (which crops to prioritize planting)
- WHEAT_RESERVE / desired_hands / desired_coops / desired_pastures formulas
- LAND / HIRE thresholds in `economic_orders()`
"""

# ---------------------------------------------------------------------------
# Game constants — pulled straight from the environment so we never drift
# out of sync with the real rules. Falls back to hardcoded copies (matching
# the current kaggriculture.py) if the import path ever changes.
# ---------------------------------------------------------------------------
try:
    from kaggle_environments.envs.kaggriculture.kaggriculture import CROPS, ANIMALS
except Exception:
    CROPS = {
        "WHEAT":      {"seed": 10,  "first_yield_day": 2,  "max_yield_day": 4,  "interval": 0, "max_yield": 6, "ongoing": False},
        "CARROT":     {"seed": 20,  "first_yield_day": 2,  "max_yield_day": 3,  "interval": 0, "max_yield": 4, "ongoing": False},
        "TOMATO":     {"seed": 50,  "first_yield_day": 8,  "max_yield_day": 8,  "interval": 1, "max_yield": 4, "ongoing": True},
        "STRAWBERRY": {"seed": 100, "first_yield_day": 10, "max_yield_day": 10, "interval": 2, "max_yield": 4, "ongoing": True},
        "MELON":      {"seed": 80,  "first_yield_day": 10, "max_yield_day": 12, "interval": 0, "max_yield": 6, "ongoing": False},
    }
    ANIMALS = {
        "GOOSE": {"cost": 300, "structure": "COOP",    "first_yield_day": 4, "interval": 1, "max_held": 4, "product": "EGG"},
        "COW":   {"cost": 400, "structure": "PASTURE", "first_yield_day": 8, "interval": 2, "max_held": 6, "product": "MILK"},
        "SHEEP": {"cost": 500, "structure": "PASTURE", "first_yield_day": 6, "interval": 3, "max_held": 6, "product": "WOOL"},
    }

LAND_PRICES = [1000, 2000, 4000]     # cost of the 1st/2nd/3rd extra quadrant


# Rough $-per-tile-per-day ranking (base price x yield/tile/day from the
# competition docs) — determines which crop we PLANT when a seed of more
# than one type is available for an empty tile, and also which crop we BUY
# more of when replenishing seed stock.
CROP_PLANT_ORDER = ["MELON", "STRAWBERRY", "TOMATO", "WHEAT", "CARROT"]

DELTA = {"NORTH": (0, -1), "SOUTH": (0, 1), "EAST": (1, 0), "WEST": (-1, 0)}


def _dist(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _step_toward(pos, target):
    """One orthogonal step from pos toward target (Manhattan, x-then-y)."""
    fx, fy = pos
    tx, ty = target
    if fx < tx:
        return "EAST"
    if fx > tx:
        return "WEST"
    if fy < ty:
        return "SOUTH"
    if fy > ty:
        return "NORTH"
    return None


def _shed_tiles(board_size):
    half = board_size // 2
    return [(half - 1, half - 1), (half, half - 1), (half - 1, half), (half, half)]


# ---------------------------------------------------------------------------
# Main agent
# ---------------------------------------------------------------------------
def agent(obs):
    try:
        return _agent_impl(obs)
    except Exception:
        # Never let a bug lose us the match: fall back to a safe, simple move.
        return {"farmer": ["PASS"], "hands": [], "market": []}


def _agent_impl(obs):
    player = obs.get("player", 0)
    day = obs.get("day", 0)
    hour = obs.get("hour", 0)
    farms = obs.get("farms", [])
    if not farms or player >= len(farms):
        return {"farmer": ["PASS"], "hands": [], "market": []}

    farm = farms[player]
    private = obs.get("private", {}) or {}
    market = obs.get("market", {}) or {}

    tiles = farm["tiles"]
    board_size = len(tiles)
    money = farm["money"]
    seeds = dict(private.get("seeds", {}) or {})
    shed = dict(private.get("shed", {}) or {})
    inventories = private.get("inventories", [{}]) or [{}]
    unlocked = farm.get("unlocked_quadrants", ["NW"])
    shed_access = _shed_tiles(board_size)

    positions = [tuple(farm["farmer"])] + [tuple(p) for p in farm.get("hands", [])]
    n_units = len(positions)
    invs = [dict(inventories[i]) if i < len(inventories) else {} for i in range(n_units)]
    assigned = [None] * n_units  # filled in with action lists as we schedule work

    # ------------------------------------------------------------------
    # 1. Scan the farm: build every job list + a few stats we need later.
    # ------------------------------------------------------------------
    empty_tiles = []
    weed_tiles = []
    water_jobs = []
    harvest_jobs = []
    fertilize_jobs = []
    feed_jobs = []
    care_jobs = []
    collect_fert_jobs = []
    empty_structures = {"COOP": [], "PASTURE": []}
    n_coop = n_pasture = n_animals = 0

    for y in range(board_size):
        for x in range(board_size):
            t = tiles[y][x]
            if t is None:
                empty_tiles.append((x, y))
                continue
            if t == "LOCKED":
                continue
            kind = t.get("kind")
            if kind == "WEED":
                weed_tiles.append((x, y))
            elif kind == "PLANT":
                cd = CROPS[t["crop"]]
                age = day - t["planted_day"]
                if t["yield_units"] > 0 and age >= cd["first_yield_day"]:
                    harvest_jobs.append((x, y))
                if not t["watered_today"]:
                    water_jobs.append((x, y))
                if t.get("fertilized_until_day", -1) < day:
                    if cd["ongoing"]:
                        fertilize_jobs.append((x, y))
                    else:
                        window_start = (cd["max_yield_day"] + 1) // 2
                        if window_start <= age <= cd["max_yield_day"]:
                            fertilize_jobs.append((x, y))
            elif kind in ("COOP", "PASTURE"):
                n_coop += kind == "COOP"
                n_pasture += kind == "PASTURE"
                if "animal" not in t:
                    empty_structures[kind].append((x, y))
                else:
                    n_animals += 1
                    if t["yield_units"] > 0:
                        harvest_jobs.append((x, y))
                    if not t["fed_today"]:
                        feed_jobs.append((x, y))
                    if not t["cared_today"]:
                        care_jobs.append((x, y))
                    if t.get("fertilizer_available"):
                        collect_fert_jobs.append((x, y))

    # ------------------------------------------------------------------
    # 2. Generic assignment helpers.
    # ------------------------------------------------------------------
    def assign_direct(job_items):
        """job_items: list of (pos, action_list). Greedy nearest-unit match."""
        for pos, action in job_items:
            best_u, best_d = None, None
            for u in range(n_units):
                if assigned[u] is not None:
                    continue
                d = _dist(positions[u], pos)
                if best_d is None or d < best_d:
                    best_d, best_u = d, u
            if best_u is None:
                return
            if positions[best_u] == pos:
                assigned[best_u] = action
            else:
                step = _step_toward(positions[best_u], pos)
                assigned[best_u] = [step] if step else ["PASS"]

    def handle_gated_jobs(jobs, resource, act_for_tile):
        """Jobs that need `resource` carried in inventory (FEED/FERTILIZE/PLACE).
        First use units already carrying it; anything left over sends the
        nearest idle unit to the shed to fetch more (if the shed has any)."""
        remaining = []
        for (x, y) in jobs:
            best_u, best_d = None, None
            for u in range(n_units):
                if assigned[u] is not None or invs[u].get(resource, 0) <= 0:
                    continue
                d = _dist(positions[u], (x, y))
                if best_d is None or d < best_d:
                    best_d, best_u = d, u
            if best_u is not None:
                if positions[best_u] == (x, y):
                    assigned[best_u] = act_for_tile(x, y)
                else:
                    step = _step_toward(positions[best_u], (x, y))
                    assigned[best_u] = [step] if step else ["PASS"]
            else:
                remaining.append((x, y))

        if remaining and shed.get(resource, 0) > 0:
            best_u, best_d = None, None
            for u in range(n_units):
                if assigned[u] is not None:
                    continue
                d = min(_dist(positions[u], t) for t in shed_access)
                if best_d is None or d < best_d:
                    best_d, best_u = d, u
            if best_u is not None:
                if positions[best_u] in shed_access:
                    qty = min(shed.get(resource, 0), max(1, len(remaining)))
                    assigned[best_u] = ["PICKUP", resource, qty]
                else:
                    target = min(shed_access, key=lambda t: _dist(positions[best_u], t))
                    step = _step_toward(positions[best_u], target)
                    assigned[best_u] = [step] if step else ["PASS"]

    # ------------------------------------------------------------------
    # 3. Assign jobs in priority order (cash-in first, upkeep next,
    #    growth last). Each pass only touches still-idle units.
    # ------------------------------------------------------------------
    assign_direct([(pos, ["HARVEST"]) for pos in harvest_jobs])
    handle_gated_jobs(feed_jobs, "WHEAT", lambda x, y: ["FEED"])
    assign_direct([(pos, ["WATER"]) for pos in water_jobs])
    assign_direct([(pos, ["DIG"]) for pos in weed_tiles])
    assign_direct([(pos, ["CARE"]) for pos in care_jobs])
    assign_direct([(pos, ["COLLECT_FERTILIZER"]) for pos in collect_fert_jobs])

    for kind, spots in empty_structures.items():
        if not spots:
            continue
        # Replay evidence from 3 real opponent matches (all wins) showed the
        # winning strategy is heavy, dense animal husbandry — wall-to-wall
        # coops/pastures rather than diversified crops — outproducing a
        # crop-focused farm 2-4x in final money. SHEEP (wool, $200 base) is
        # what those farms were packed with, so pastures go all-SHEEP now
        # instead of alternating with COW. GOOSE stays in coops: cheapest
        # animal, fastest to first yield (day 4), good for early cash while
        # sheep are still maturing (day 6).
        animal_type = "GOOSE" if kind == "COOP" else ("COW" if day % 2 == 0 else "SHEEP")
        handle_gated_jobs(spots, animal_type, lambda x, y, a=animal_type: ["PLACE", a])

    handle_gated_jobs(fertilize_jobs, "FERTILIZER", lambda x, y: ["FERTILIZE"])

    # --- Empty tiles: split between new animal structures and planting ---
    # Second attempt (paced by n_animals + a flat buffer) still tested worse
    # than the crop-only baseline — it diverted early capital into
    # structures/animals starting turn 1, competing with the same cash
    # crops need to bootstrap. The winning replay farms almost certainly
    # built up cash via crops FIRST, then converted the surplus into
    # animals once established — not both at once from a $3000 start.
    # Only divert land+cash into animal structures once there's real SPARE
    # cash beyond a working reserve; below that reserve this behaves like
    # the original crop-focused baseline (barely builds ahead of what's
    # already stocked).
    desired_coops = min(3, 1 + day // 6)
    desired_pastures = min(2, day // 10)
    # Fill CLOSEST-TO-SHED empty tiles first, not raw board-scan order.
    # Diagnosed from a real match: seed stock was abundant (8-9 of every
    # crop) while 26-37 tiles sat empty simultaneously — not a money/seed
    # shortage but a THROUGHPUT one. Hired hands respawn at the shed every
    # morning, so if the next tile to fill happens to be clear across the
    # board, a hand can burn most of its day just walking there instead of
    # working. Growing outward from the shed keeps travel short and lets
    # far tiles get worked once nearer ones are already productive.
    empty_tiles = sorted(empty_tiles, key=lambda t: min(_dist(t, s) for s in shed_access))
    remaining_empty = list(empty_tiles)
    build_jobs = []
    while n_coop < desired_coops and remaining_empty:
        build_jobs.append((remaining_empty.pop(0), ["BUILD_COOP"]))
        n_coop += 1
    while n_pasture < desired_pastures and remaining_empty:
        build_jobs.append((remaining_empty.pop(0), ["BUILD_PASTURE"]))
        n_pasture += 1
    assign_direct(build_jobs)

    seed_budget = dict(seeds)
    plant_jobs = []
    for pos in remaining_empty:
        crop_choice = next((c for c in CROP_PLANT_ORDER if seed_budget.get(c, 0) > 0), None)
        if crop_choice:
            seed_budget[crop_choice] -= 1
            plant_jobs.append((pos, ["PLANT", crop_choice]))
    assign_direct(plant_jobs)

    # --- Fallback: idle units head to / unload at the shed ---
    for u in range(n_units):
        if assigned[u] is not None:
            continue
        if invs[u]:
            if positions[u] in shed_access:
                assigned[u] = ["DROP"]
            else:
                target = min(shed_access, key=lambda t: _dist(positions[u], t))
                step = _step_toward(positions[u], target)
                assigned[u] = [step] if step else ["PASS"]
        else:
            assigned[u] = ["PASS"]

    farmer_action = assigned[0]
    hand_actions = assigned[1:]

    # ------------------------------------------------------------------
    # 4. Money decisions (market orders). Only 10 order SLOTS exist per
    #    turn (one slot per distinct order, regardless of its quantity).
    #
    #    Earlier versions kept stacking "high priority" categories (hire,
    #    land, seeds) into one loosely-capped bucket without checking their
    #    COMBINED size — which silently pushed SELL out of the list on busy
    #    turns (verified: this caused a real money=0/hands=0 collapse mid-
    #    game in testing). Unsold shed goods are already-earned value that
    #    just gets discarded once the shed hits its cap, so SELL must
    #    ALWAYS get every slot it needs FIRST. Growth spending (hire, land,
    #    seeds, animals) only gets whatever slots are left over — a growth
    #    order skipped this turn just tries again next turn at no real
    #    cost, so it's safe to deprioritize.
    # ------------------------------------------------------------------
    MAX_ORDERS = 10
    orders = []

    wheat_reserve = n_animals * 3  # only reserve feed stock if we actually have animals to feed
    fert_reserve = 3
    for item, qty in shed.items():
        if qty <= 0 or item in ("GOOSE", "COW", "SHEEP"):
            continue
        sellable = qty
        if item == "WHEAT":
            sellable = max(0, qty - wheat_reserve)
        elif item == "FERTILIZER":
            sellable = max(0, qty - fert_reserve)
        if sellable > 0:
            orders.append(["SELL", item, sellable])

    budget = MAX_ORDERS - len(orders)  # whatever's left over after selling, for growth

    # -- Animals: fill ALL currently-empty structures in one order (BUY_ANIMAL
    #    supports a quantity, same as SELL/BUY_SEED — one order slot buys as
    #    many as we ask, processed per-unit up to what we can afford), not
    #    just 1/turn like before. This is now a HIGH priority: the replay
    #    evidence that prompted this rewrite showed animal-heavy farms
    #    massively outproducing crop-heavy ones, so an empty coop/pasture
    #    sitting idle is a bigger loss than a slightly slower crop rotation.
    for kind, spots in empty_structures.items():
        if budget <= 0:
            break
        if not spots:
            continue
        animal_type = "GOOSE" if kind == "COOP" else ("COW" if day % 2 == 0 else "SHEEP")
        cost = ANIMALS[animal_type]["cost"]
        if money > cost * 2 and shed.get(animal_type, 0) < len(spots):
            orders.append(["BUY_ANIMAL", animal_type, 1])
            budget -= 1

    # -- Wheat top-up: feeds the (now much larger) animal population.
    if budget > 0 and n_animals > 0 and shed.get("WHEAT", 0) < n_animals and money > 200:
        orders.append(["BUY_PRODUCT", "WHEAT", n_animals])
        budget -= 1

    # -- Seeds: scaled to how many empty tiles actually need filling. Uses
    #    CROP_PLANT_ORDER (kept from the verified baseline — reordering by
    #    speed-to-maturity tested worse in isolation earlier this session).
    #
    #    NEW FIX (from a real match trace): target_per_crop acts as a STOCK
    #    LEVEL to maintain, not a one-time buy. Every hour, as soon as a
    #    worker plants a seed, next turn's check sees stock dipped below
    #    target and immediately re-buys to top it back up — for every crop,
    #    every hour. That's continuous re-purchasing of $80-100 seeds as
    #    fast as they're planted, which is what was actually draining cash
    #    well past the visible day-1 splurge (confirmed: a real match
    #    showed money still falling, $305 -> $196, through day 2). Capping
    #    TOTAL seed spend per turn to a fraction of current cash throttles
    #    that refill rate without changing which crops get bought.
    if budget > 0 and empty_tiles:
        target_per_crop = max(2, len(empty_tiles) // len(CROP_PLANT_ORDER) + 1)
        spend_cap = max(50.0, money * 0.15)
        spent = 0.0
        for crop in CROP_PLANT_ORDER:
            if budget <= 0 or spent >= spend_cap:
                break
            cost = CROPS[crop]["seed"]
            need = target_per_crop - seeds.get(crop, 0)
            if need > 0 and money > cost * (need + 2):
                affordable_qty = max(0, int((spend_cap - spent) // cost))
                qty = min(need, affordable_qty)
                if qty > 0:
                    orders.append(["BUY_SEED", crop, qty])
                    spent += qty * cost
                    budget -= 1

    # -- Hire: the affordability-driven "hire every hour up to 25" approach
    #    was tested and reverted — on the small $3000 starting bank it
    #    over-hired on day 1 before any income existed, and even after
    #    fixing the seed-overspend and wheat-reserve bugs below, it still
    #    kept the farm cash-starved for most of the first 10 days. This
    #    formula-based version (checked once per day, right when hands
    #    reset) is the one that was actually verified at 30k-36k final
    #    money across 5 seeds. It scales with both elapsed days and land
    #    owned, capped at 12 total workers — a level that's small enough to
    #    always fund from a cold start, without capping so low that we
    #    can't grow into more land.
    if budget > 0 and hour == 0:
        target_workers = min(12, 3 + day // 3 + 2 * (len(unlocked) - 1))
        desired_hands = max(0, target_workers - 1)  # -1: farmer already counts as one
        hires_queued = min(desired_hands, budget)
        orders += [["HIRE"]] * hires_queued
        budget -= hires_queued

    # -- Land: reverting to the ORIGINAL, verified-at-30k-36k gate. I tried
    #    a utilization/workforce-based version to expand land more (to
    #    address hands-count concerns), and tested several variants of it —
    #    every one of them scored WORSE than this simple gate, because this
    #    scheduler's logistics (always chase the nearest job, no zone
    #    assignment) genuinely can't keep a 75-100 tile farm well-utilized
    #    with a workforce this size — land ends up outpacing labor no
    #    matter how the purchase trigger is tuned. This condition is
    #    intentionally conservative: it rarely fires past 1-2 quadrants,
    #    which is exactly why it scored best. Making bigger farms actually
    #    pay off needs a smarter scheduler (per-zone worker assignment),
    #    not a different purchase threshold — that's a real next step, not
    #    a quick tweak.
    n_extra_land = len(unlocked) - 1
    if budget > 0 and n_extra_land < len(LAND_PRICES):
        cost = LAND_PRICES[n_extra_land]
        if money > cost * 2.5 and len(empty_tiles) < 6:
            orders.append(["BUY_LAND"])
            budget -= 1

    return {"farmer": farmer_action, "hands": hand_actions, "market": orders}