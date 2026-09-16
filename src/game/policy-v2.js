import { bullHeads, cheapestRowIndex, rowPenalty, targetRowIndex } from './engine.js'

export const POLICY_V2_FORMAT = 'ntw-policy-v2'
export const POLICY_V2_STATE_SIZE = 420
export const POLICY_V2_ACTION_SIZE = 17

function clamp01(value) {
  return Math.max(0, Math.min(1, value))
}

function dense(input, layer, activate = true) {
  const output = layer.bias.map((bias, row) => {
    let value = bias
    const weights = layer.weight[row]
    for (let column = 0; column < input.length; column += 1) value += weights[column] * input[column]
    return activate ? value / (1 + Math.exp(-value)) : value
  })
  return output
}

export function forwardLayers(input, layers, finalActivation = false) {
  return layers.reduce((values, layer, index) => dense(values, layer, index < layers.length - 1 || finalActivation), input)
}

export function validatePolicyV2(model) {
  if (model?.format !== POLICY_V2_FORMAT) throw new Error(`不支持的策略模型格式：${model?.format || 'unknown'}`)
  if (model.featureVersion !== 2 || model.stateSize !== POLICY_V2_STATE_SIZE || model.actionSize !== POLICY_V2_ACTION_SIZE) {
    throw new Error(`策略特征契约不兼容：v${model.featureVersion}/${model.stateSize}/${model.actionSize}`)
  }
  if (model.cards !== 54) throw new Error('Policy v2 当前仅支持五人 10n+4 的 54 张牌')
  if (!model.stateLayers?.length || !model.actionLayers?.length || !model.valueLayers?.length) throw new Error('Policy v2 缺少网络层')
  return true
}

/** Encode public information only. `playedCards` is ordered [self, opponent 1, ...]. */
export function encodePolicyV2State(observation) {
  if (observation.deckSize !== 54 || observation.playerCount !== 5) throw new Error('Policy v2 只支持五人 10n+4')
  const handMask = Array(54).fill(0)
  observation.hand.forEach((card) => { handMask[card - 1] = 1 })
  const seenMask = Array(54).fill(0)
  observation.seenCards.forEach((card) => { seenMask[card - 1] = 1 })
  const playedMasks = Array.from({ length: 5 }, () => Array(54).fill(0))
  ;(observation.playedCards || []).slice(0, 5).forEach((cards, player) => {
    cards.forEach((card) => { playedMasks[player][card - 1] = 1 })
  })
  const board = observation.rows.flatMap((row) => {
    const values = row.map((card) => card / 54)
    while (values.length < 6) values.push(0)
    return values
  })
  const lengths = observation.rows.map((row) => row.length / 6)
  const costs = observation.rows.map((row) => clamp01(rowPenalty(row) / 25))
  const tails = observation.rows.map((row) => row[row.length - 1] / 54)
  const scores = [...(observation.scores || Array(5).fill(0))]
  while (scores.length < 5) scores.push(0)
  const features = [
    ...handMask,
    ...seenMask,
    ...playedMasks.flat(),
    ...board,
    ...lengths,
    ...costs,
    ...tails,
    ...scores.slice(0, 5).map((score) => clamp01(score / 50)),
    (10 - observation.hand.length) / 10,
  ]
  if (features.length !== POLICY_V2_STATE_SIZE) throw new Error(`Policy v2 状态长度错误：${features.length}`)
  return features
}

export function encodePolicyV2Action(observation, card) {
  if (!observation.hand.includes(card)) throw new Error(`Policy v2 收到非法候选牌 ${card}`)
  let rowIndex = targetRowIndex(observation.rows, card)
  const tooLow = rowIndex < 0
  if (tooLow) rowIndex = cheapestRowIndex(observation.rows)
  const row = observation.rows[rowIndex]
  const tail = row[row.length - 1]
  const captures = tooLow || row.length >= 5
  const immediate = captures ? rowPenalty(row) : 0
  const sorted = [...observation.hand].sort((left, right) => left - right)
  const rank = sorted.indexOf(card) / Math.max(1, sorted.length - 1)
  const rowOneHot = Array(4).fill(0)
  rowOneHot[rowIndex] = 1
  const known = new Set([...observation.seenCards, ...observation.hand])
  let unknown = 0
  let interval = 0
  for (let candidate = 1; candidate <= 54; candidate += 1) {
    if (known.has(candidate)) continue
    unknown += 1
    if (!tooLow && candidate > tail && candidate < card) interval += 1
  }
  const intervalFraction = interval / Math.max(1, unknown)
  const minTail = Math.min(...observation.rows.map((candidateRow) => candidateRow[candidateRow.length - 1]))
  const trapped = observation.hand.filter((candidate) => candidate < minTail).length / 10
  const features = [
    card / 54,
    bullHeads(card) / 7,
    rank,
    ...rowOneHot,
    tooLow ? 0 : clamp01((card - tail - 1) / 54),
    tooLow ? 0 : clamp01((5 - row.length) / 5),
    clamp01(immediate / 25),
    tooLow ? 1 : 0,
    captures ? 1 : 0,
    intervalFraction,
    intervalFraction,
    trapped,
    clamp01(rowPenalty(row) / 25),
    clamp01(row.length / 5),
  ]
  if (features.length !== POLICY_V2_ACTION_SIZE) throw new Error(`Policy v2 动作长度错误：${features.length}`)
  return features
}

export function evaluatePolicyV2Cards(observation, model) {
  validatePolicyV2(model)
  const context = forwardLayers(encodePolicyV2State(observation), model.stateLayers, true)
  const value = forwardLayers(context, model.valueLayers)[0]
  return observation.hand.map((card) => {
    const action = encodePolicyV2Action(observation, card)
    const logit = forwardLayers([...context, ...action], model.actionLayers)[0]
    return { card, utility: -logit, logit, value, risk: 0, disasterRate: 0 }
  }).sort((left, right) => left.utility - right.utility || left.card - right.card)
}

export function choosePolicyV2Card(observation, model) {
  const evaluations = evaluatePolicyV2Cards(observation, model)
  return { card: evaluations[0].card, evaluations, policyV2: true, value: evaluations[0].value }
}

export function validateCompactPolicyV2(model) {
  if (model?.format !== 'ntw-policy-v2-compact') throw new Error(`不支持的紧凑策略模型格式：${model?.format || 'unknown'}`)
  if (model.featureVersion !== 2 || model.stateSize !== POLICY_V2_STATE_SIZE || model.actionSize !== POLICY_V2_ACTION_SIZE || model.cards !== 54) {
    throw new Error(`紧凑策略特征契约不兼容：v${model.featureVersion}/${model.stateSize}/${model.actionSize}`)
  }
  if (!model.stateLayers?.length || !model.actionLayers?.length) throw new Error('紧凑策略缺少网络层')
  return true
}

export function evaluateCompactPolicyV2Cards(observation, model) {
  validateCompactPolicyV2(model)
  const context = forwardLayers(encodePolicyV2State(observation), model.stateLayers, true)
  return observation.hand.map((card) => {
    const logit = forwardLayers([...context, ...encodePolicyV2Action(observation, card)], model.actionLayers)[0]
    return { card, utility: -logit, logit, risk: 0, disasterRate: 0 }
  }).sort((left, right) => left.utility - right.utility || left.card - right.card)
}

export function chooseCompactPolicyV2Card(observation, model) {
  const evaluations = evaluateCompactPolicyV2Cards(observation, model)
  return { card: evaluations[0].card, evaluations, compactPolicyV2: true }
}

export function evaluateCompactOpponentCards(observation, model) {
  const contextualV1 = model?.format === 'ntw-contextual-opponent-v1'
  const contextualV2 = model?.format === 'ntw-contextual-opponent-v2'
  const contextual = contextualV1 || contextualV2
  if ((!contextual && (model?.format !== 'ntw-compact-opponent-v1' || model.actionSize !== POLICY_V2_ACTION_SIZE)) || (contextualV1 && model.featureSize !== 44) || (contextualV2 && model.featureSize !== 143) || !model.layers?.length) {
    throw new Error('紧凑对手模型格式不兼容')
  }
  const contextualFeatures = (card) => {
    const hand = observation.hand
    const mean = hand.reduce((sum, value) => sum + value, 0) / hand.length
    const variance = hand.reduce((sum, value) => sum + (value - mean) ** 2, 0) / hand.length
    const scores = [...(observation.scores || Array(5).fill(0))]
    while (scores.length < 5) scores.push(0)
    const opponents = scores.slice(1)
    const scoreRank = opponents.filter((value) => value < scores[0]).length / 4
    const rowContext = observation.rows.flatMap((row) => [row[row.length - 1] / 54, row.length / 5, clamp01(rowPenalty(row) / 25)])
    return [
      ...encodePolicyV2Action(observation, card),
      (10 - hand.length) / 10,
      hand.length / 10,
      Math.min(...hand) / 54,
      Math.max(...hand) / 54,
      mean / 54,
      Math.sqrt(variance) / 54,
      hand.reduce((sum, value) => sum + bullHeads(value), 0) / hand.length / 7,
      clamp01(scores[0] / 50),
      clamp01(Math.min(...opponents) / 50),
      clamp01(opponents.reduce((sum, value) => sum + value, 0) / opponents.length / 50),
      clamp01(Math.max(...opponents) / 50),
      scoreRank,
      ...rowContext,
      Math.min(...observation.rows.map((row) => row[row.length - 1])) / 54,
      Math.max(...observation.rows.map((row) => row[row.length - 1])) / 54,
      observation.seenCards.length / 54,
    ]
  }
  const contextualFeaturesV2 = (card) => {
    const handMask = Array(54).fill(0)
    observation.hand.forEach((value) => { handMask[value - 1] = 1 })
    const seenMask = Array(54).fill(0)
    observation.seenCards.forEach((value) => { seenMask[value - 1] = 1 })
    const rowContext = observation.rows.flatMap((row) => [row[row.length - 1] / 54, row.length / 5, clamp01(rowPenalty(row) / 25)])
    const scores = [...(observation.scores || Array(5).fill(0))]
    while (scores.length < 5) scores.push(0)
    return [
      ...encodePolicyV2Action(observation, card),
      ...handMask,
      ...seenMask,
      ...rowContext,
      ...scores.slice(0, 5).map((score) => clamp01(score / 50)),
      (10 - observation.hand.length) / 10,
    ]
  }
  return observation.hand.map((card) => ({
    card,
    logit: forwardLayers(contextualV2 ? contextualFeaturesV2(card) : contextual ? contextualFeatures(card) : encodePolicyV2Action(observation, card), model.layers)[0],
  })).sort((left, right) => right.logit - left.logit || left.card - right.card)
}

export function compactOpponentLogProbability(observation, model, card, temperature = 1) {
  const evaluations = evaluateCompactOpponentCards(observation, model)
  const maximum = evaluations[0].logit
  const weights = evaluations.map((entry) => Math.exp((entry.logit - maximum) / Math.max(0.05, temperature)))
  const total = weights.reduce((sum, value) => sum + value, 0)
  const index = evaluations.findIndex((entry) => entry.card === card)
  return index < 0 ? -20 : Math.log(Math.max(1e-9, weights[index] / total))
}

export function chooseCompactOpponentCard(observation, model, rng, temperature = 0) {
  const evaluations = evaluateCompactOpponentCards(observation, model)
  if (!rng || temperature <= 0) return evaluations[0].card
  const maximum = evaluations[0].logit
  const weights = evaluations.map((entry) => Math.exp((entry.logit - maximum) / temperature))
  const total = weights.reduce((sum, value) => sum + value, 0)
  let roll = rng() * total
  for (let index = 0; index < evaluations.length; index += 1) {
    roll -= weights[index]
    if (roll <= 0) return evaluations[index].card
  }
  return evaluations[evaluations.length - 1].card
}
