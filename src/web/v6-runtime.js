// Vite transforms this JSON import into JavaScript in both dev and builds.
// Keeping it behind the router's dynamic import avoids loading V6 on GPU routes.
import model from '../../artifacts/small-player-exploration/v6/model.json'
import { chooseSmallCard } from '../../scripts/small-strategy-runtime.mjs'

export const chooseV6 = observation => chooseSmallCard(observation, model)
