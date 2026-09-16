import { encodeNeuralCandidate } from '../src/game/neural-features.js'

const permutations = (items) => items.length ? items.flatMap((x, i) => permutations(items.filter((_, j) => j !== i)).map(p => [x, ...p])) : [[]]
const rowPermutations = permutations([0, 1, 2, 3])
const basisIndices = [264, 266, 265, 267, 269, 255, 254, 257, 256]

function forward(features, model) {
  let values = features
  for (let index = 0; index < model.layers.length; index++) {
    const layer = model.layers[index]
    values = layer.bias.map((bias, row) => {
      let value = bias
      for (let j = 0; j < values.length; j++) value += values[j] * layer.weight[row][j]
      return index === model.layers.length - 1 ? value : Math.max(0, value)
    })
  }
  return values[0] * 10
}

export function permuteSmallFeatures(features, permutation) {
  const result = features.slice()
  for (let i = 0; i < 4; i++) {
    for (let j = 0; j < 6; j++) result[208 + i * 6 + j] = features[208 + permutation[i] * 6 + j]
    for (let j = 0; j < 3; j++) result[232 + i * 3 + j] = features[232 + permutation[i] * 3 + j]
    result[258 + i] = features[258 + permutation[i]]
  }
  return result
}

export function evaluateSmallCards(observation, model) {
  const p = observation.playerCount
  if (![2, 3, 4].includes(p) || observation.deckSize !== 104) throw new Error('Small strategy requires 2-4 players and 104 cards')
  if (model.format !== 'ntw-small-player-v1') throw new Error('Unknown small strategy format')
  return observation.hand.map(card => {
    const features = encodeNeuralCandidate(observation, card)
    let utility
    if (model.kind === 'specialist') utility = forward(features, model.models[p])
    else {
      utility = forward(features, model.base)
      if (model.kind === 'symmetry') {
        const mean = rowPermutations.reduce((sum, perm) => sum + forward(permuteSmallFeatures(features, perm), model.base), 0) / rowPermutations.length
        const blend = model.settings[p]
        utility = (1 - blend) * utility + blend * mean
      } else if (model.kind === 'residual') {
        const x = basisIndices.map(i => features[i])
        const phase = features[251]
        const deficit = Math.max(-1, Math.min(1, (features[244] - features[245]) * 3))
        const basis = [...x, ...x.map(v => v * phase), ...x.map(v => v * deficit)]
        utility += basis.reduce((sum, value, i) => sum + value * model.settings[p][i], 0)
      } else throw new Error('Unknown small strategy kind')
    }
    return { card, utility }
  }).sort((a, b) => a.utility - b.utility || a.card - b.card)
}

export function chooseSmallCard(observation, model) {
  const evaluations = evaluateSmallCards(observation, model)
  return { card: evaluations[0].card, evaluations }
}
