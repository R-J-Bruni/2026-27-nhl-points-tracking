"""Fetch NHL standings + remaining schedules, simulate the rest of the season, and
write data/standings.json.

How a team's projection is built
--------------------------------
1. TEAM TALENT (points per game it "really" earns), blending three things:
     - actual points per game so far,
     - goal-difference-based (Pythagorean) points per game, which strips out luck,
     - the pre-season line (the market's view of the roster), which counts like
       K_PRIOR games of evidence and fades as the season goes on.
2. INJURIES: data/injuries.json lists key players who are out, for how many more
   games, and how much each one is worth in points per game. Those amounts are taken
   off the team's talent for exactly the games they will miss.
3. SIMULATION: every remaining game in the league is played SIMS times using each
   team's talent (with home-ice advantage and realistic overtime/shootout points).
   Each simulated season also gets its own error in how good every team really is,
   because that is only an estimate. The projection is the average final total;
   "p_over" is the share of simulated seasons in which the team finishes above its
   pre-season line.
"""
import json, math, random, sys, time, urllib.request, datetime, pathlib

BASE = "https://api-web.nhle.com/v1"
ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "standings.json"
PICKS = ROOT / "data" / "picks.json"
INJURIES = ROOT / "data" / "injuries.json"

SEASON_GAMES = 84   # 2026-27 is an 84-game season
K_PRIOR = 24        # the pre-season line counts like this many games of evidence
LUCK_WEIGHT = 0.5   # share of "current form" taken from goal difference vs. actual points
PYTH_EXP = 2.0      # Pythagorean exponent for goals
HOME_LOGIT = 0.18   # home-ice advantage on the log-odds scale (~54.5% for an even matchup)
SIMS = 10000
TALENT_SD = 0.115   # real spread in team talent, in points per game (about 9.5 points over 82 games)


def logit(p):
    return math.log(p / (1 - p))


def log5(a, b):
    """Chance team A beats team B given each one's win-share vs. an average opponent."""
    return a * (1 - b) / (a * (1 - b) + b * (1 - a))


def to_share(rate, q):
    """Win chance vs. an average team that makes a team earn `rate` points per game, given a share q
    of games reach overtime (the loser then still gets 1 point). Inverts E = p*(2-0.35q) + 0.675q."""
    return (rate - 0.675 * q) / (2 - 0.35 * q)


def team_rates(rows, lines):
    """-> (rate per team in points/game, pool = points handed out per game, both teams)."""
    tot_gp = sum(r["gp"] for r in rows.values())
    tot_pts = sum(r["pts"] for r in rows.values())
    pool = 2 * tot_pts / tot_gp
    avg_rate = pool / 2
    rate = {}
    for k, r in rows.items():
        g = r["gp"]
        prior = lines[k] / SEASON_GAMES if lines.get(k) else avg_rate
        if g == 0:
            rate[k] = prior
            continue
        obs = r["pts"] / g
        gf, ga = r.get("gf"), r.get("ga")
        if gf is not None and ga is not None and gf + ga > 0:
            wp = gf ** PYTH_EXP / (gf ** PYTH_EXP + ga ** PYTH_EXP)
            skill = (1 - LUCK_WEIGHT) * obs + LUCK_WEIGHT * pool * wp
        else:
            skill = obs
        rate[k] = (g * skill + K_PRIOR * prior) / (g + K_PRIOR)
    return rate, pool


def analytic_model(rows, remaining_opps, lines):
    """Fallback with no simulation or injuries (used if schedules can't be loaded).
    remaining_opps: abbr -> [opponent abbr] or None."""
    rate, pool = team_rates(rows, lines)
    share = {k: min(0.95, max(0.05, v / pool)) for k, v in rate.items()}
    out = {}
    for k, r in rows.items():
        left = SEASON_GAMES - r["gp"]
        opps = remaining_opps.get(k)
        if opps and abs(len(opps) - left) <= 1 and all(o in share for o in opps):
            exp = sum(pool * log5(share[k], share[o]) for o in opps)
            src = "schedule"
        else:
            exp = left * rate[k]
            src = "rate"
        out[k] = {
            "proj": round(r["pts"] + exp, 1),
            "pace": round(r["pts"] / r["gp"] * SEASON_GAMES, 1) if r["gp"] else None,
            "p_over": None,
            "src": src,
        }
    return out


def simulate(rows, sched, lines, injuries=None, sims=SIMS, seed=None):
    """sched: abbr -> ordered list of (game_id, home, away) still to play.
    injuries: abbr -> [{"games_out": int, "impact": points-per-game}, ...]."""
    injuries = injuries or {}
    rng = random.Random(seed)
    rnd = rng.random
    rate, pool = team_rates(rows, lines)
    q = min(0.30, max(0.15, pool - 2))          # share of games that reach OT / shootout

    abbrs = sorted(rows)
    idx = {k: i for i, k in enumerate(abbrs)}

    def impact(k, j):
        return sum(i["impact"] for i in injuries.get(k, []) if j < i["games_out"])

    games, pos = {}, {}
    for k, lst in sched.items():
        for j, (gid, h, a) in enumerate(lst):
            games[gid] = (h, a)
            pos[(k, gid)] = j

    pre = []
    for gid, (h, a) in games.items():
        rh = max(0.3, rate[h] - impact(h, pos.get((h, gid), 10 ** 6)))
        ra = max(0.3, rate[a] - impact(a, pos.get((a, gid), 10 ** 6)))
        sh = min(0.95, max(0.05, to_share(rh, q)))
        sa = min(0.95, max(0.05, to_share(ra, q)))
        p = 1 / (1 + math.exp(-(logit(sh) - logit(sa) + HOME_LOGIT)))   # home win chance
        p_ot = 0.5 + 0.35 * (p - 0.5)                                    # luckier-than-skill in OT
        p_reg = min(0.99, max(0.01, (p - q * p_ot) / (1 - q)))           # keeps overall win chance = p
        pre.append((idx[h], idx[a], p_reg, p_ot))

    # We only ESTIMATE each team's talent, and the estimate is shakier early in the season.
    # Give every simulated season its own talent error so the probabilities aren't overconfident.
    talent_sd = [TALENT_SD * math.sqrt(K_PRIOR / (rows[k]["gp"] + K_PRIOR)) * len(sched.get(k, [])) for k in abbrs]
    base = [rows[k]["pts"] for k in abbrs]
    line = [lines.get(k) for k in abbrs]
    n = len(abbrs)
    total = [0.0] * n
    over = [0] * n
    for _ in range(sims):
        pts = base[:]
        for ih, ia, pr, po in pre:
            if rnd() < q:                   # overtime / shootout
                if rnd() < po:
                    pts[ih] += 2; pts[ia] += 1
                else:
                    pts[ia] += 2; pts[ih] += 1
            elif rnd() < pr:
                pts[ih] += 2
            else:
                pts[ia] += 2
        for i in range(n):
            final = pts[i] + (rng.gauss(0, talent_sd[i]) if talent_sd[i] else 0)
            total[i] += final
            if line[i] is not None and final > line[i]:
                over[i] += 1

    out = {}
    for k in abbrs:
        i = idx[k]
        r = rows[k]
        out[k] = {
            "proj": round(total[i] / sims, 1),
            "pace": round(r["pts"] / r["gp"] * SEASON_GAMES, 1) if r["gp"] else None,
            "p_over": round(over[i] / sims, 3) if line[i] is not None else None,
            "src": "sim",
        }
    return out


def get(path):
    last = None
    for i in range(3):
        try:
            req = urllib.request.Request(BASE + path, headers={"User-Agent": "nhl-tracker"})
            with urllib.request.urlopen(req, timeout=30) as r:   # follows the redirect
                return json.load(r)
        except Exception as e:
            last = e
            time.sleep(2)
    raise last


def load_injuries():
    """data/injuries.json -> abbr -> [{"games_out", "impact"}]. Bad entries are skipped."""
    try:
        raw = json.loads(INJURIES.read_text())
    except Exception:
        return {}
    out = {}
    for k, lst in (raw.get("teams") or {}).items():
        for e in lst or []:
            try:
                out.setdefault(k, []).append({
                    "games_out": int(e.get("games_out", 0)),
                    "impact": float(e.get("impact", 0)),
                })
            except (TypeError, ValueError):
                print("skipping bad injury entry for", k, e)
    return out


def main():
    data = get("/standings/now")
    rows = {}
    for t in data["standings"]:
        rows[t["teamAbbrev"]["default"]] = {
            "gp": t["gamesPlayed"], "pts": t["points"],
            "gf": t.get("goalFor"), "ga": t.get("goalAgainst"),
        }
    if len(rows) != 32:
        sys.exit(f"Expected 32 teams, got {len(rows)}; leaving file unchanged")

    lines = {k: v.get("line") for k, v in json.loads(PICKS.read_text())["teams"].items()}

    sched, opps_only, ok = {}, {}, True
    for k in rows:
        try:
            s = get(f"/club-schedule-season/{k}/now")
            lst = []
            for g in s.get("games", []):
                if g.get("gameType") != 2 or g.get("gameState") not in ("FUT", "PRE"):
                    continue
                lst.append((g["id"], g["homeTeam"]["abbrev"], g["awayTeam"]["abbrev"]))
            sched[k] = lst
            opps_only[k] = [a if h == k else h for _, h, a in lst]
            if abs(len(lst) - (SEASON_GAMES - rows[k]["gp"])) > 1:
                ok = False
        except Exception as e:
            print("schedule failed for", k, e)
            ok = False
        time.sleep(0.3)

    injuries = load_injuries()
    if ok and len(sched) == 32:
        model = simulate(rows, sched, lines, injuries)
        print(f"Simulated {SIMS} seasons; injuries applied to {len(injuries)} teams")
    else:
        print("Schedules incomplete -> falling back to the simple model (no simulation)")
        model = analytic_model(rows, opps_only, lines)

    teams = {k: {"gp": r["gp"], "pts": r["pts"], **model[k]} for k, r in rows.items()}
    out = {"updated": datetime.date.today().isoformat(), "method": "model",
           "sims": SIMS if ok else 0, "teams": teams}
    OUT.write_text(json.dumps(out, indent=1))
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
