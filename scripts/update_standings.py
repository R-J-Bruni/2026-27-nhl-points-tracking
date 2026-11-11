"""Fetch current NHL standings and write data/standings.json."""
import json, sys, urllib.request, datetime, pathlib

URL = "https://api-web.nhle.com/v1/standings/now"
OUT = pathlib.Path(__file__).resolve().parent.parent / "data" / "standings.json"

req = urllib.request.Request(URL, headers={"User-Agent": "nhl-tracker"})
with urllib.request.urlopen(req, timeout=30) as r:   # follows the redirect
    data = json.load(r)

teams = {}
for t in data["standings"]:
    teams[t["teamAbbrev"]["default"]] = {"gp": t["gamesPlayed"], "pts": t["points"]}

if len(teams) != 32:
    sys.exit(f"Expected 32 teams, got {len(teams)}; leaving file unchanged")

out = {"updated": datetime.date.today().isoformat(), "teams": teams}
OUT.write_text(json.dumps(out, indent=1))
print("Updated", len(teams), "teams")
