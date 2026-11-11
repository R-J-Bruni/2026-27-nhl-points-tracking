# NHL Over/Under Tracker

Static site (GitHub Pages). `data/picks.json` holds the lines and picks (O = Over, U = Under; fill in Ben's "b" values).
`data/standings.json` is refreshed daily by the GitHub Action in `.github/workflows/update.yml`.
The page also pulls live standings from the NHL each time it opens, falling back to that file.
