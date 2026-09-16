import { bullHeads, cheapestRowIndex, rowPenalty, targetRowIndex } from './engine.js'

export const NEURAL_MODEL_FORMAT = 'ntw-neural-v1'
export const NEURAL_FEATURE_VERSION = 1
export const NEURAL_FEATURE_SIZE = 270

function clamp01(value) {
  return Math.max(0, Math.min(1, value))
}

function placementFeatures(observation, card) {
  const { rows, hand, deckSize, playerCount, seenCards } = observation
  let rowIndex = targetRowIndex(rows, card)
  const tooLow = rowIndex < 0
  if (tooLow) rowIndex = cheapestRowIndex(rows)
  const row = rows[rowIndex]
  const tail = row[row.length - 1]
  const captures = tooLow || row.length >= 5
  const immediate = captures ? rowPenalty(row) : 0
  const gap = tooLow ? 0 : Math.max(0, card - tail - 1)
  const openSlots = tooLow ? 0 : Math.max(0, 5 - row.length)

  const known = new Set([...seenCards, ...hand])
  let unknown = 0
  let interval = 0
  for (let candidate = 1; candidate <= deckSize; candidate += 1) {
    if (known.has(candidate)) continue
    unknown += 1
    if (!tooLow && candidate > tail && candidate < card) interval += 1
  }
  const intervalFraction = interval / Math.max(1, unknown)
  const traffic = intervalFraction * Math.max(0, playerCount - 1)
  const minTail = Math.min(...rows.map((candidateRow) => candidateRow[candidateRow.length - 1]))
  const trapped = hand.filter((candidate) => candidate !== card && candidate < minTail).length
  const sortedHand = [...hand].sort((left, right) => left - right)
  const rank = sortedHand.indexOf(card)
  const leftGap = rank > 0 ? card - sortedHand[rank - 1] : card
  const rightGap = rank + 1 < sortedHand.length ? sortedHand[rank + 1] - card : deckSize + 1 - card
  const rowOneHot = Array(4).fill(0)
  rowOneHot[rowIndex] = 1

  return [
    card / 104,
    card / deckSize,
    bullHeads(card) / 7,
    rank / Math.max(1, sortedHand.length - 1),
    leftGap / 104,
    rightGap / 104,
    ...rowOneHot,
    gap / 104,
    openSlots / 5,
    immediate / 25,
    tooLow ? 1 : 0,
    captures ? 1 : 0,
    intervalFraction,
    traffic / 9,
    trapped / 9,
  ].map(clamp01)
}

/** Encode only information available to the acting player. */
export function encodeNeuralCandidate(observation, card) {
  if (!observation.hand.includes(card)) throw new Error(`神经模型收到非法候选牌 ${card}`)
  if (observation.rows.length !== 4) throw new Error('神经模型要求固定 4 行')
  if (observation.deckSize > 104) throw new Error('神经模型最多支持 104 张牌')

  const handMask = Array(104).fill(0)
  observation.hand.forEach((value) => { handMask[value - 1] = 1 })
  const seenMask = Array(104).fill(0)
  observation.seenCards.forEach((value) => { seenMask[value - 1] = 1 })
  const board = observation.rows.flatMap((row) => {
    const values = row.map((value) => value / 104)
    while (values.length < 6) values.push(0)
    return values
  })
  const rowSummaries = observation.rows.flatMap((row) => [
    row.length / 5,
    clamp01(rowPenalty(row) / 25),
    row[row.length - 1] / 104,
  ])
  const scores = observation.scores || Array(observation.playerCount).fill(0)
  const ownScore = scores[0] || 0
  const otherScores = scores.slice(1)
  const scoreSummaries = [
    clamp01(ownScore / 50),
    clamp01((otherScores.length ? Math.min(...otherScores) : 0) / 50),
    clamp01((otherScores.reduce((sum, value) => sum + value, 0) / Math.max(1, otherScores.length)) / 50),
    clamp01((otherScores.length ? Math.max(...otherScores) : 0) / 50),
  ]
  const globals = [
    observation.deckSize / 104,
    observation.playerCount / 10,
    observation.hand.length / 10,
    (10 - observation.hand.length) / 10,
  ]
  const features = [
    ...handMask,
    ...seenMask,
    ...board,
    ...rowSummaries,
    ...scoreSummaries,
    ...globals,
    ...placementFeatures(observation, card),
  ]
  if (features.length !== NEURAL_FEATURE_SIZE) throw new Error(`神经特征长度错误：${features.length}`)
  return features
}

function dense(input, layer, activation) {
  const output = layer.bias.map((bias, row) => {
    let value = bias
    const weights = layer.weight[row]
    for (let column = 0; column < input.length; column += 1) value += weights[column] * input[column]
    return value
  })
  if (activation === 'relu') return output.map((value) => Math.max(0, value))
  if (activation === 'tanh') return output.map(Math.tanh)
  return output
}

function forward(input, model) {
  return model.layers.reduce((values, layer, index) => (
    dense(values, layer, index === model.layers.length - 1 ? null : model.activation)
  ), input)
}

export function validateNeuralModel(model) {
  if (model?.format !== NEURAL_MODEL_FORMAT) throw new Error(`不支持的神经模型格式：${model?.format || 'unknown'}`)
  if (model.featureVersion !== NEURAL_FEATURE_VERSION || model.featureSize !== NEURAL_FEATURE_SIZE) {
    throw new Error(`神经特征契约不兼容：v${model.featureVersion}/${model.featureSize}`)
  }
  if (!Array.isArray(model.layers) || !model.layers.length) throw new Error('神经模型没有全连接层')
  return true
}

export function evaluateNeuralCards(observation, model) {
  validateNeuralModel(model)
  const scale = model.targetScale || 10
  return observation.hand.map((card) => ({
    card,
    utility: forward(encodeNeuralCandidate(observation, card), model)[0] * scale,
    risk: 0,
    disasterRate: 0,
  })).sort((left, right) => left.utility - right.utility || left.card - right.card)
}

export function chooseNeuralCard(observation, model) {
  const evaluations = evaluateNeuralCards(observation, model)
  return { card: evaluations[0].card, evaluations, neural: true }
}
