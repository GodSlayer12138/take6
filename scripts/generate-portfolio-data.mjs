import { createWriteStream, mkdirSync, readFileSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { once } from 'node:events'

import { createGame } from '../src/game/engine.js'
import { mulberry32, playArenaGame, setPolicyV2ModelForEvaluation } from '../src/game/ai-core.js'

function argument(name, fallback) {
  const index = process.argv.indexOf(`--${name}`)
  return index >= 0 ? process.argv[index + 1] : fallback
}

async function writeLine(stream, value) {
  if (!stream.write(`${JSON.stringify(value)}\n`)) await once(stream, 'drain')
}

const games = Number(argument('games', '300'))
const baseSeed = Number(argument('seed', '221000000'))
const output = resolve(argument('output', 'artifacts/training/portfolio.jsonl'))
const samples = Number(argument('samples', '18'))
const modelPath = resolve(argument('model', 'artifacts/models/ntw-policy-v2-distill2-t120-i050.json'))
const strategies = argument('strategies', '').split(',').filter(Boolean)
if (strategies.length < 2) throw new Error('--strategies requires at least two comma-separated strategies')
setPolicyV2ModelForEvaluation(JSON.parse(readFileSync(modelPath, 'utf8')))
const opponents = ['champion', 'external_mcs', 'cautious', 'random']

function play(strategy, gameIndex, gameSeed) {
  const focalSeat = (5 - (gameIndex % 5)) % 5
  const seats = Array(5)
  seats[focalSeat] = strategy
  opponents.forEach((opponent, offset) => { seats[(focalSeat + offset + 1) % 5] = opponent })
  const result = playArenaGame({
    playerCount: 5,
    deckMode: 'adaptive',
    strategies: seats,
    seed: gameSeed,
    championSamples: samples,
    legacySamples: Math.max(8, Math.round(samples / 2)),
  })
  const score = result.scores[focalSeat]
  const minimum = Math.min(...result.scores)
  const winnerCount = result.scores.filter((value) => value === minimum).length
  const win = score === minimum ? 1 / winnerCount : 0
  const lower = result.scores.filter((value) => value < score).length
  const tied = result.scores.filter((value) => value === score).length
  const rank = 1 + lower + (tied - 1) / 2
  return { win, score, rank, reward: win * 8 - (rank - 1) * 0.8 - score * 0.12 }
}

mkdirSync(dirname(output), { recursive: true })
const stream = createWriteStream(output, { encoding: 'utf8' })
await writeLine(stream, { meta: { format: 'ntw-portfolio-v1', games, baseSeed, samples, modelPath, strategies } })
const started = performance.now()
for (let gameIndex = 0; gameIndex < games; gameIndex += 1) {
  const gameSeed = baseSeed + gameIndex * 65537
  const focalSeat = (5 - (gameIndex % 5)) % 5
  const deal = createGame(5, 'adaptive', mulberry32(gameSeed))
  const rows = deal.rows.map((row) => [...row])
  const hand = [...deal.hands[focalSeat]]
  const results = strategies.map((strategy) => play(strategy, gameIndex, gameSeed))
  await writeLine(stream, {
    gameIndex,
    seed: gameSeed,
    playerSeat: focalSeat,
    rows,
    hand,
    seenCards: rows.flat(),
    deckSize: deal.deckSize,
    playerCount: 5,
    scores: [0, 0, 0, 0, 0],
    playedCards: [[], [], [], [], []],
    results,
  })
  const reportEvery = Math.max(1, Math.floor(games / 10))
  if ((gameIndex + 1) % reportEvery === 0) {
    console.log(`[portfolio] ${gameIndex + 1}/${games} games · ${((performance.now() - started) / 1000).toFixed(1)}s`)
  }
}
stream.end()
await once(stream, 'finish')
console.log(`[portfolio] wrote ${output}`)
