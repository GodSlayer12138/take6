import { bullHeads, cheapestRowIndex, rowPenalty, targetRowIndex } from './engine.js'

export const RL6NIMMT_FORMAT = 'rl-6-nimmt-browser-v1'

function seededRandom(seed) {
  let state = seed >>> 0
  return () => {
    state += 0x6d2b79f5
    let value = state
    value = Math.imul(value ^ (value >>> 15), value | 1)
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61)
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296
  }
}

function shuffle(items, rng) {
  const result = [...items]
  for (let index = result.length - 1; index > 0; index -= 1) {
    const other = Math.floor(rng() * (index + 1))
    ;[result[index], result[other]] = [result[other], result[index]]
  }
  return result
}

function cloneRows(rows) {
  return rows.map((row) => [...row])
}

function removeCard(hand, card) {
  const index = hand.indexOf(card)
  if (index >= 0) hand.splice(index, 1)
}

function resolveCards(rows, cards) {
  const nextRows = cloneRows(rows)
  const penalties = Array(cards.length).fill(0)
  const ordered = cards.map((card, player) => ({ card, player })).sort((a, b) => a.card - b.card)
  for (const { card, player } of ordered) {
    let rowIndex = targetRowIndex(nextRows, card)
    const tooLow = rowIndex < 0
    if (tooLow) rowIndex = cheapestRowIndex(nextRows)
    if (tooLow || nextRows[rowIndex].length >= 5) {
      penalties[player] += rowPenalty(nextRows[rowIndex])
      nextRows[rowIndex] = [card]
    } else {
      nextRows[rowIndex].push(card)
    }
  }
  return { rows: nextRows, penalties }
}

/** Converts our 1-based public observation to the repository's raw 47-value state. */
export function encodeRl6NimmtState(observation) {
  if (observation.rows.length !== 4) throw new Error('rl-6-nimmt 模型要求固定 4 行')
  const hand = [...observation.hand].sort((a, b) => a - b).map((card) => card - 1)
  while (hand.length < 10) hand.push(-1)
  const cardsPerRow = observation.rows.map((row) => row.length)
  const highestPerRow = observation.rows.map((row) => row[row.length - 1] - 1)
  const scorePerRow = observation.rows.map((row) => row.reduce((sum, card) => sum + bullHeads(card), 0))
  const board = observation.rows.flatMap((row) => {
    const encoded = row.map((card) => card - 1)
    while (encoded.length < 6) encoded.push(-1)
    return encoded
  })
  return [...hand, observation.playerCount, ...cardsPerRow, ...highestPerRow, ...scorePerRow, ...board]
}

function normalizeRange(value, min, max) {
  return -1 + 2 * (value - min) / (max - min)
}

/** Exact port of SechsNimmtStateNormalization from the PyTorch project. */
export function normalizeRl6NimmtInput(rawState, { action = null, cards = 104, rows = 4 } = {}) {
  const output = []
  if (action != null) output.push(normalizeRange(action - 1, 0, cards - 1))
  let position = 0
  for (const value of rawState.slice(position, position + 10)) output.push(normalizeRange(value, 0, cards - 1))
  position += 10
  output.push(normalizeRange(rawState[position], 0, 6))
  position += 1
  for (const value of rawState.slice(position, position + rows)) output.push(normalizeRange(value, 1, 5))
  position += rows
  for (const value of rawState.slice(position, position + rows)) output.push(normalizeRange(value, 0, cards - 1))
  position += rows
  for (const value of rawState.slice(position, position + rows)) output.push(normalizeRange(value, 1, 10))
  position += rows
  for (const value of rawState.slice(position)) output.push(normalizeRange(value, 0, cards - 1))
  return output
}

function dense(input, layer, activation) {
  const output = layer.bias.map((bias, row) => {
    let value = bias
    for (let column = 0; column < input.length; column += 1) value += layer.weight[row][column] * input[column]
    return value
  })
  if (activation === 'relu') return output.map((value) => Math.max(0, value))
  if (activation === 'tanh') return output.map(Math.tanh)
  return output
}

function forwardLayers(input, layers, activation = 'relu') {
  return layers.reduce((values, layer, index) => dense(values, layer, index === layers.length - 1 ? null : activation), input)
}

export function validateConvertedModel(model) {
  if (model?.format !== RL6NIMMT_FORMAT) throw new Error(`不支持的模型格式：${model?.format || 'unknown'}`)
  if (!['action-conditioned-policy', 'action-values'].includes(model.architecture)) throw new Error(`不支持的网络结构：${model.architecture}`)
  if (!Array.isArray(model.layers) || model.layers.length === 0) throw new Error('模型没有可用的全连接层')
  return true
}

/** Runs converted PolicyMCS/PUCT policy heads or DQN-style action value heads. */
export function chooseConvertedModelCard(observation, model) {
  validateConvertedModel(model)
  const rawState = encodeRl6NimmtState(observation)
  let ranked
  if (model.architecture === 'action-conditioned-policy') {
    ranked = observation.hand.map((card) => ({
      card,
      value: forwardLayers(normalizeRl6NimmtInput(rawState, { action: card, cards: model.cards || 104 }), model.layers, model.activation)[0],
    }))
  } else {
    const values = forwardLayers(normalizeRl6NimmtInput(rawState, { cards: model.cards || 104 }), model.layers, model.activation)
    ranked = observation.hand.map((card) => ({ card, value: values[card - 1] }))
  }
  ranked.sort((left, right) => right.value - left.value || left.card - right.card)
  return { card: ranked[0].card, evaluations: ranked.map((entry) => ({ card: entry.card, utility: -entry.value, risk: 0, disasterRate: 0 })) }
}

/**
 * Browser port of the public repository's weight-free MCSAgent. The original
 * uses up to 100 uniformly random full-game rollouts total; we balance that
 * budget across root actions to remove accidental zero-visit actions.
 */
export function chooseExternalMcsCard(observation, { mcMax = 100 } = {}) {
  const rng = seededRandom(observation.seed || 0x726c366e)
  const known = new Set([...observation.seenCards, ...observation.hand])
  const available = []
  for (let card = 1; card <= observation.deckSize; card += 1) if (!known.has(card)) available.push(card)
  const rolloutsPerCard = Math.max(1, Math.floor(mcMax / observation.hand.length))
  const evaluations = observation.hand.map((rootCard) => {
    let penaltySum = 0
    let disasterCount = 0
    const outcomes = []
    for (let rollout = 0; rollout < rolloutsPerCard; rollout += 1) {
      const shuffled = shuffle(available, rng)
      const hands = [[...observation.hand]]
      for (let opponent = 1; opponent < observation.playerCount; opponent += 1) {
        hands.push(shuffled.slice((opponent - 1) * observation.hand.length, opponent * observation.hand.length))
      }
      let rows = cloneRows(observation.rows)
      let focalPenalty = 0
      while (hands[0].length) {
        const cards = hands.map((hand, player) => {
          if (player === 0 && hand.length === observation.hand.length) return rootCard
          return hand[Math.floor(rng() * hand.length)]
        })
        cards.forEach((card, player) => removeCard(hands[player], card))
        const result = resolveCards(rows, cards)
        rows = result.rows
        focalPenalty += result.penalties[0]
      }
      outcomes.push(focalPenalty)
      penaltySum += focalPenalty
      if (focalPenalty >= 8) disasterCount += 1
    }
    const mean = penaltySum / rolloutsPerCard
    const variance = outcomes.reduce((sum, value) => sum + (value - mean) ** 2, 0) / rolloutsPerCard
    return { card: rootCard, utility: mean, expectedPenalty: mean, risk: Math.sqrt(variance), disasterRate: disasterCount / rolloutsPerCard }
  }).sort((left, right) => left.utility - right.utility || left.card - right.card)
  return { card: evaluations[0].card, evaluations, sampleCount: rolloutsPerCard * observation.hand.length }
}
