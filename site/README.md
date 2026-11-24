# Demo site

An interactive site for the results. It uses Next.js, Recharts and an SVG map in km coordinates. "ACO" on the site is the ACO with the tuned parameters (`ACOParams()` defaults).

- **Mission replay** (`/`): the recorded 24 h Camp Fire missions. Select 6, 10 or 12 drones and two planners. Greedy and VNS are available at 10 drones only. The maps show the strongest pheromone links that each strategy keeps when the map changes.
- **Change size** (`/explore`): the controlled-change experiment. Set the share of changed points (5–60 %), the locality and the fire stage.

## Develop

```sh
cd site
npm install
npm run dev
npm run typecheck
npm run build
```

## Data

The files in `public/data/` come from the experiment outputs. Run these commands from the repository root.

```sh
python scripts/site/record_pheromone.py outputs/site_pheromone
python scripts/site/export_missions.py site/public/data
python scripts/site/change_grid.py outputs/site_change --trials 12 --workers 9
cp outputs/site_change/change_grid.json site/public/data/
```

`export_missions.py` needs `outputs/tuning_missions_d10/stages/` from `scripts/run_tuned_missions.py`. The replay shows one mission (seed 0) for each planner. The means on the replay page use all 10 seeds.

## Deploy

```sh
cd site
vercel --prod
```
