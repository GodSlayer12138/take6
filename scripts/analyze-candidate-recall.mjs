import { createReadStream, readFileSync } from 'node:fs'
import { createInterface } from 'node:readline'
import { resolve } from 'node:path'

import { chooseCard } from '../src/game/ai-core.js'
import { evaluateNeuralCards } from '../src/game/neural-features.js'
import { evaluatePolicyV2Cards } from '../src/game/policy-v2.js'
import legacyModel from '../src/game/models/ntw-champion.json' with { type: 'json' }

const dataPath = resolve(process.argv[2])
const modelPath = resolve(process.argv[3])
const policyModel = JSON.parse(readFileSync(modelPath, 'utf8'))

const specifications = []
for (const keep of [3, 4, 5]) {
  specifications.push({ id: `v2-${keep}`, keep, mode: 'v2', weight: 0 })
  for (const weight of [0, 0.25, 0.5, 0.75]) {
    specifications.push({ id: `blend-${Math.round(weight * 100)}-${keep}`, keep, mode: 'blend', weight })
    specifications.push({ id: `union-${Math.round(weight * 100)}-${keep}`, keep, mode: 'union', weight })
  }
  specifications.push({ id: `split-${keep}`, keep, mode: 'split', weight: 0 })
}

const stats = new Map(specifications.map((specification) => [specification.id, {
  specification,
  hit: 0,
  count: 0,
  byTurn: Array.from({ length: 10 }, () => ({ hit: 0, count: 0 })),
}]))

function select(specification, observation, v2, legacy, heuristicCard) {
  const v2Rank = new Map(v2.map((entry, index) => [entry.card, index]))
  const legacyRank = new Map(legacy.map((entry, index) => [entry.card, index]))
  const blended = [...v2].map((entry) => ({
    card: entry.card,
    rank: v2Rank.get(entry.card) * (1 - specification.weight) + legacyRank.get(entry.card) * specification.weight,
  })).sort((left, right) => left.rank - right.rank || left.card - right.card)

  let cards
  if (specification.mode === 'v2') {
    cards = v2.slice(0, specification.keep).map((entry) => entry.card)
  } else if (specification.mode === 'blend') {
    cards = blended.slice(0, specification.keep).map((entry) => entry.card)
  } else if (specification.mode === 'union') {
    const each = Math.max(1, Math.ceil(specification.keep / 2))
    cards = [...new Set([
      ...blended.slice(0, each).map((entry) => entry.card),
      ...legacy.slice(0, each).map((entry) => entry.card),
    ])].slice(0, specification.keep)
    for (const entry of blended) {
      if (cards.length >= specification.keep) break
      if (!cards.includes(entry.card)) cards.push(entry.card)
    }
  } else {
    const each = Math.max(1, Math.ceil(specification.keep / 2))
    cards = [...new Set([
      ...v2.slice(0, each).map((entry) => entry.card),
      ...legacy.slice(0, each).map((entry) => entry.card),
    ])].slice(0, specification.keep)
    for (const entry of v2) {
      if (cards.length >= specification.keep) break
      if (!cards.includes(entry.card)) cards.push(entry.card)
    }
  }
  if (!cards.includes(heuristicCard)) cards[cards.length - 1] = heuristicCard
  return cards
}

const input = createInterface({ input: createReadStream(dataPath, { encoding: 'utf8' }), crlfDelay: Infinity })
for await (const line of input) {
  const record = JSON.parse(line)
  if (record.meta) continue
  const observation = {
    rows: record.rows,
    hand: record.hand,
    seenCards: record.seenCards,
    deckSize: record.deckSize,
    playerCount: record.playerCount,
    scores: record.scores,
    playedCards: record.playedCards,
    seed: 1,
  }
  const teacher = [...record.labels].sort((left, right) => left[1] - right[1] || left[0] - right[0])[0][0]
  const v2 = evaluatePolicyV2Cards(observation, policyModel)
  const legacy = evaluateNeuralCards(observation, legacyModel)
  const heuristicCard = chooseCard(observation, 'cautious').card
  for (const specification of specifications) {
    const selected = select(specification, observation, v2, legacy, heuristicCard)
    const entry = stats.get(specification.id)
    const hit = selected.includes(teacher) ? 1 : 0
    entry.hit += hit
    entry.count += 1
    entry.byTurn[record.turn].hit += hit
    entry.byTurn[record.turn].count += 1
  }
}

const results = [...stats.values()].map((entry) => ({
  id: entry.specification.id,
  recall: entry.hit / entry.count,
  byTurn: entry.byTurn.map(({ hit, count }) => hit / count),
})).sort((left, right) => right.recall - left.recall || left.id.localeCompare(right.id))

console.log(JSON.stringify({ dataPath, modelPath, states: results.length ? stats.values().next().value.count : 0, results }, null, 2))
