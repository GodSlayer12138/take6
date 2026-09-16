import { createReadStream, readFileSync } from 'node:fs'
import { createInterface } from 'node:readline'
import { resolve } from 'node:path'

import { chooseCard, setPolicyV2ModelForEvaluation } from '../src/game/ai-core.js'

const dataPath = resolve(process.argv[2])
const modelPath = resolve(process.argv[3])
const strategy = process.argv[4]
const gameLimit = Number(process.argv[5] || 200)
const samples = Number(process.argv[6] || 18)

if (!strategy) throw new Error('strategy is required')
setPolicyV2ModelForEvaluation(JSON.parse(readFileSync(modelPath, 'utf8')))

let meta = null
const games = new Map()
const input = createInterface({ input: createReadStream(dataPath, { encoding: 'utf8' }), crlfDelay: Infinity })
for await (const line of input) {
  const record = JSON.parse(line)
  if (record.meta) {
    meta = record.meta
    continue
  }
  if (record.gameIndex >= gameLimit) continue
  if (!games.has(record.gameIndex)) games.set(record.gameIndex, [])
  games.get(record.gameIndex).push(record)
}

let regret = 0
let squaredRegret = 0
let exact = 0
let decisions = 0
const byTurn = Array.from({ length: 10 }, () => ({ regret: 0, exact: 0, count: 0 }))
const started = performance.now()

for (const [gameIndex, records] of games) {
  records.sort((left, right) => left.turn - right.turn)
  const focalPlayer = gameIndex % 5
  const gameSeed = meta.baseSeed + gameIndex * 65537
  for (const record of records) {
    const history = Array.from({ length: record.turn }, (_, round) => ({
      rows: records[round].rows.map((row) => [...row]),
      seenCards: [...records[round].seenCards],
      ownCard: record.playedCards[0][round],
      opponentCards: record.playedCards.slice(1).map((cards) => cards[round]),
    }))
    const observation = {
      rows: record.rows.map((row) => [...row]),
      hand: [...record.hand],
      seenCards: [...record.seenCards],
      deckSize: record.deckSize,
      playerCount: record.playerCount,
      scores: [...record.scores],
      playedCards: record.playedCards.map((cards) => [...cards]),
      opponentStrategies: ['champion', 'external_mcs', 'cautious', 'random'],
      history,
      samples,
      seed: gameSeed + record.turn * 104729 + focalPlayer * 8191,
    }
    const labels = new Map(record.labels)
    const teacher = [...record.labels].sort((left, right) => left[1] - right[1] || left[0] - right[0])[0]
    const selected = chooseCard(observation, strategy).card
    const selectedRegret = labels.get(selected) - teacher[1]
    regret += selectedRegret
    squaredRegret += selectedRegret ** 2
    exact += selected === teacher[0] ? 1 : 0
    decisions += 1
    byTurn[record.turn].regret += selectedRegret
    byTurn[record.turn].exact += selected === teacher[0] ? 1 : 0
    byTurn[record.turn].count += 1
  }
}

const meanRegret = regret / decisions
const variance = decisions > 1 ? (squaredRegret - decisions * meanRegret ** 2) / (decisions - 1) : 0
console.log(JSON.stringify({
  dataPath,
  modelPath,
  strategy,
  games: games.size,
  decisions,
  samples,
  meanRegret,
  regretStandardError: Math.sqrt(Math.max(0, variance) / decisions),
  exactRate: exact / decisions,
  byTurn: byTurn.map((entry) => ({
    regret: entry.regret / entry.count,
    exactRate: entry.exact / entry.count,
  })),
  seconds: (performance.now() - started) / 1000,
}))
