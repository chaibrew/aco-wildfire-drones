# Demo site

An interactive site for the ACO drone-routing results. It is built on a Next.js dashboard template (monospace Geist, dot-matrix map, stat cards that flip to an explanation), with Recharts charts and an SVG map in km coordinates. Light and dark mode follow the system setting.

"ACO" on the site means the ACO with the parameters chosen by F-Race and irace (`ACOParams()` defaults). The untuned runs are left out.

Drones are drawn with the 8-frame pixel-art quadcopter sprite in `public/sprites/drone-0..7.png`, cut from the sprite sheet at its native size with the background made transparent. The rotors cycle while the replay plays and each drone turns to face its direction of travel.

- **Mission replay** (`/`). The recorded 24 h Camp Fire missions, animated. Pick 6, 10 or 12 drones and any two planners: the four ACO pheromone strategies, plus greedy and VNS at 10 drones. Play, pause (space bar), change speed (2 min/s to 2 h/s) and scrub through time. Monitoring points stay grey until they are due soon, overdue or missed. The ACO maps draw the strongest pheromone links each strategy keeps when the map changes; these are recorded by `scripts/site/record_pheromone.py`, which re-runs the seed-0 missions (the export checks the routes match the original recordings).
- **Change size** (`/explore`). The controlled-change experiment. A slider sets the share of points changed (5–60%), and toggles set the locality and the fire state. The map shows the points, the changed and removed points, the pheromone right after the change and after replanning, and each planner's routes.
- **Conclusion** (`/conclusion`). Each claim from the conclusion, with numbers computed in the browser from the same data, and links that open the view that shows it.

## Develop

```sh
cd site
npm install
npm run dev        # http://localhost:3000
npm run typecheck  # tsc
npm run build      # production build to .next/
```

## Data

`public/data/` is generated from the experiment outputs:

```sh
# mission replays (needs outputs/tuning_missions*/stages/*.pkl from scripts/run_tuned_missions.py)
python scripts/site/export_missions.py site/public/data

# controlled-change grid: 2 fire stages x 12 trials x 6 sizes x 2 localities (~3 min on 9 cores)
python scripts/site/change_grid.py outputs/site_change --trials 12 --workers 9
cp outputs/site_change/change_grid.json site/public/data/
```

The replay exports one mission (seed 0) for every planner, to keep the site small (about 4 MB). The per-planner means on the replay page still use all 10 seeds, from `missions.csv`.

## Deploy

```sh
cd site
vercel          # preview deployment
vercel --prod   # production
```

Vercel detects Next.js and runs `npm run build`. Pages keep their settings in the query string (for example `/?d=12&a=tuned_retention&b=tuned_full_reset&t=6`), so any view can be shared as a link.
