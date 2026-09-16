import { createWriteStream, mkdirSync, readFileSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { once } from 'node:events'

import { chooseCard, mulberry32, setPolicyV2ModelForEvaluation } from '../src/game/ai-core.js'
import { createGame, resolveTurn } from '../src/game/engine.js'

function argument(name, fallback) {
  const index = process.argv.indexOf(`--${name}`)
  return index >= 0 ? process.argv[index + 1] : fallback
}

function removeCard(hand, card) {
  const index = hand.indexOf(card)
  if (index < 0) throw new Error(`cannot remove card ${card}`)
  hand.splice(index, 1)
}

async function writeLine(stream, value) {
  if (!stream.write(`${JSON.stringify(value)}\n`)) await once(stream, 'drain')
}

const games = Number(argument('games', '1200'))
const output = resolve(argument('output', 'artifacts/training/arena-distillation.jsonl'))
const baseSeed = Number(argument('seed', '340100000'))
const teacherSamples = Number(argument('samples', '54'))
const modelPath = resolve(argument('policy-v2-model', 'artifacts/models/ntw-policy-v2-onpolicy2-b05-e020.json'))
const teacherStrategy = argument('teacher-strategy', 'neural_hybrid_v2_k100_crn_ex4_cp5_cf_bm8_p0')
const model = JSON.parse(readFileSync(modelPath, 'utf8'))
setPolicyV2ModelForEvaluation(model)
const fixedOpponents = ['champion', 'external_mcs', 'cautious', 'random']

mkdirSync(dirname(output), { recursive: true })
const stream = createWriteStream(output, { encoding: 'utf8' })
await writeLine(stream, { meta: { format: 'ntw-arena-distillation-v1', games, baseSeed, teacherSamples, modelPath, teacherStrategy } })
const started = performance.now()
let states = 0
let labels = 0

for (let gameIndex = 0; gameIndex < games; gameIndex += 1) {
  const gameSeed = baseSeed + gameIndex * 65537
  const focalPlayer = gameIndex % 5
  const strategies = Array(5)
  strategies[focalPlayer] = teacherStrategy
  fixedOpponents.forEach((strategy, index) => { strategies[(focalPlayer + index + 1) % 5] = strategy })
  const deal = createGame(5, 'adaptive', mulberry32(gameSeed))
  const hands = deal.hands.map((hand) => [...hand])
  const scores = Array(5).fill(0)
  let rows = deal.rows.map((row) => [...row])
  const seenCards = [...deal.initialCards]
  const publicHistory = []

  for (let turn = 0; turn < 10; turn += 1) {
    const actions = []
    for (let player = 0; player < 5; player += 1) {
      const relativePlayers = Array.from({ length: 5 }, (_, offset) => (player + offset) % 5)
      const observation = {
        rows: rows.map((row) => [...row]),
        hand: [...hands[player]],
        seenCards: [...seenCards],
        deckSize: deal.deckSize,
        playerCount: 5,
        playerSeat: player,
        scores: relativePlayers.map((seat) => scores[seat]),
        playedCards: relativePlayers.map((seat) => publicHistory.map((round) => round[seat])),
        opponentStrategies: relativePlayers.slice(1).map((seat) => strategies[seat]),
        samples: player === focalPlayer ? teacherSamples : 18,
        seed: gameSeed + turn * 104729 + player * 8191,
      }
      const decision = chooseCard(observation, strategies[player])
      if (player === focalPlayer) {
        await writeLine(stream, {
          gameIndex,
          player: 0,
          turn,
          deckMode: 'adaptive',
          rows: observation.rows,
          hand: observation.hand,
          seenCards: observation.seenCards,
          deckSize: observation.deckSize,
          playerCount: observation.playerCount,
          playerSeat: observation.playerSeat,
          scores: observation.scores,
          playedCards: observation.playedCards,
          labels: decision.evaluations.map(({ card, utility }) => [card, utility]),
        })
        states += 1
        labels += decision.evaluations.length
      }
      actions.push({ playerId: player, card: decision.card })
    }

    publicHistory.push(actions.map(({ card }) => card))
    actions.forEach(({ playerId, card }) => removeCard(hands[playerId], card))
    const result = resolveTurn({ rows, actions, humanPlayerId: -1 })
    if (result.needsChoice) throw new Error('distillation arena cannot wait for a row choice')
    rows = result.rows
    actions.forEach(({ card }) => seenCards.push(card))
    actions.forEach(({ playerId }) => { scores[playerId] += result.penalties[playerId] })
  }

  const reportEvery = Math.max(1, Math.floor(games / 20))
  if ((gameIndex + 1) % reportEvery === 0 || gameIndex + 1 === games) {
    console.log(`[arena-distill] ${gameIndex + 1}/${games} games · ${states} states · ${labels} labels · ${((performance.now() - started) / 1000).toFixed(1)}s`)
  }
}

stream.end()
await once(stream, 'finish')
console.log(`[arena-distill] wrote ${output}`)
