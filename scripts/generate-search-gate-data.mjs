import { createReadStream, createWriteStream, mkdirSync, readFileSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { once } from 'node:events'
import { createInterface } from 'node:readline'

import { chooseCard, setPolicyV2ModelForEvaluation } from '../src/game/ai-core.js'
import { evaluateCompactOpponentCards } from '../src/game/policy-v2.js'

function argument(name, fallback = '') {
  const index = process.argv.indexOf(`--${name}`)
  return index >= 0 ? process.argv[index + 1] : fallback
}

async function writeLine(stream, value) {
  if (!stream.write(`${JSON.stringify(value)}\n`)) await once(stream, 'drain')
}

const inputPath = resolve(argument('input'))
const outputPath = resolve(argument('output'))
const policyPath = resolve(argument('policy'))
const counterfactualPath = resolve(argument('counterfactual'))
const strategy = argument('strategy')
const gameLimit = Number(argument('games', '100'))
if (!inputPath || !outputPath || !strategy) throw new Error('input, output, policy, counterfactual, and strategy are required')

const policy = JSON.parse(readFileSync(policyPath, 'utf8'))
const counterfactual = JSON.parse(readFileSync(counterfactualPath, 'utf8'))
setPolicyV2ModelForEvaluation(policy)

let meta = null
const games = new Map()
const input = createInterface({ input: createReadStream(inputPath, { encoding: 'utf8' }), crlfDelay: Infinity })
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
if (!meta) throw new Error('counterfactual metadata is missing')

mkdirSync(dirname(outputPath), { recursive: true })
const output = createWriteStream(outputPath, { encoding: 'utf8' })
await writeLine(output, { meta: { format: 'ntw-search-gate-v1', source: inputPath, policyPath, counterfactualPath, strategy, games: games.size } })
const started = performance.now()
let states = 0
for (const [gameIndex, records] of games) {
  records.sort((left, right) => left.turn - right.turn)
  const focalPlayer = gameIndex % 5
  const gameSeed = Number(meta.baseSeed) + gameIndex * 65537
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
      playerSeat: record.playerSeat,
      scores: [...record.scores],
      playedCards: record.playedCards.map((cards) => [...cards]),
      opponentStrategies: ['champion', 'external_mcs', 'cautious', 'random'],
      history,
      samples: 18,
      seed: gameSeed + record.turn * 104729 + focalPlayer * 8191,
    }
    const search = chooseCard(observation, strategy)
    const counterfactualEvaluations = evaluateCompactOpponentCards(observation, counterfactual)
    await writeLine(output, {
      ...record,
      searchCard: search.card,
      searchMargin: search.evaluations.length > 1 ? search.evaluations[1].utility - search.evaluations[0].utility : 10,
      counterfactualCard: counterfactualEvaluations[0].card,
      counterfactualMargin: counterfactualEvaluations.length > 1 ? counterfactualEvaluations[0].logit - counterfactualEvaluations[1].logit : 10,
    })
    states += 1
  }
  if (games.size >= 10 && states % Math.max(10, Math.floor(games.size / 10) * 10) === 0) {
    console.log(`[search-gate-data] ${states} states · ${((performance.now() - started) / 1000).toFixed(1)}s`)
  }
}
output.end()
await once(output, 'finish')
console.log(`[search-gate-data] wrote ${outputPath} states=${states}`)
