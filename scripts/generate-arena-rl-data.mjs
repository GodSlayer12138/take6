import { createWriteStream, mkdirSync, readFileSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { once } from 'node:events'
import { chooseCard, mulberry32, setPolicyV2ModelForEvaluation } from '../src/game/ai-core.js'
import { createGame, resolveTurn } from '../src/game/engine.js'
import { evaluateNeuralCards } from '../src/game/neural-features.js'

function argument(name, fallback) {
  const index = process.argv.indexOf(`--${name}`)
  return index >= 0 ? process.argv[index + 1] : fallback
}

function removeCard(hand, card) {
  const index = hand.indexOf(card)
  if (index < 0) throw new Error(`无法从手牌移除 ${card}`)
  hand.splice(index, 1)
}

async function writeLine(stream, value) {
  if (!stream.write(`${JSON.stringify(value)}\n`)) await once(stream, 'drain')
}

function sampleEvaluation(evaluations, temperature, rng) {
  const minimum = Math.min(...evaluations.map((entry) => entry.utility))
  const weights = evaluations.map((entry) => Math.exp(-(entry.utility - minimum) / temperature))
  const total = weights.reduce((sum, value) => sum + value, 0)
  let roll = rng() * total
  for (let index = 0; index < evaluations.length; index += 1) {
    roll -= weights[index]
    if (roll <= 0) return { card: evaluations[index].card, probability: weights[index] / total }
  }
  const last = evaluations.length - 1
  return { card: evaluations[last].card, probability: weights[last] / total }
}

const games = Number(argument('games', '1200'))
const output = resolve(argument('output', 'artifacts/training/arena-rl.jsonl'))
const baseSeed = Number(argument('seed', '2026082201'))
const deckMode = argument('mode', 'adaptive') === 'classic' ? 'classic' : 'adaptive'
const modelPath = resolve(argument('model', 'src/game/models/ntw-adaptive-rl.json'))
const temperature = Number(argument('temperature', '0.8'))
const samples = Number(argument('samples', '18'))
const behaviorStrategy = argument('behavior-strategy', '')
const policyV2ModelPath = argument('policy-v2-model', '')
const recordAll = argument('record-all', 'false') === 'true'
const model = JSON.parse(readFileSync(modelPath, 'utf8'))
if (policyV2ModelPath) setPolicyV2ModelForEvaluation(JSON.parse(readFileSync(resolve(policyV2ModelPath), 'utf8')))
const opponentStrategies = ['champion', 'external_mcs', 'cautious', 'random']

mkdirSync(dirname(output), { recursive: true })
const stream = createWriteStream(output, { encoding: 'utf8' })
await writeLine(stream, { meta: { format: 'ntw-arena-policy-v1', games, baseSeed, deckMode, modelPath, temperature, samples, behaviorStrategy: behaviorStrategy || null } })
const started = performance.now()
let states = 0

for (let gameIndex = 0; gameIndex < games; gameIndex += 1) {
  const gameSeed = baseSeed + gameIndex * 65537
  const focalPlayer = gameIndex % 5
  const rng = mulberry32(gameSeed ^ 0x726c6172)
  const deal = createGame(5, deckMode, mulberry32(gameSeed))
  const hands = deal.hands.map((hand) => [...hand])
  const scores = Array(5).fill(0)
  let rows = deal.rows.map((row) => [...row])
  const seenCards = [...deal.initialCards]
  const trajectory = []
  const publicHistory = []

  for (let turn = 0; turn < 10; turn += 1) {
    const actions = []
    for (let player = 0; player < 5; player += 1) {
      const relativePlayers = Array.from({ length: 5 }, (_, offset) => (player + offset) % 5)
      const perspectiveScores = relativePlayers.map((seat) => scores[seat])
      const observation = {
        rows: rows.map((row) => [...row]),
        hand: [...hands[player]],
        seenCards: [...seenCards],
        deckSize: deal.deckSize,
        playerCount: 5,
        playerSeat: player,
        scores: perspectiveScores,
        playedCards: relativePlayers.map((seat) => publicHistory.map((round) => round[seat])),
        opponentStrategies: relativePlayers.slice(1).map((seat) => (
          seat === focalPlayer ? behaviorStrategy || 'neural_rl' : opponentStrategies[((seat - focalPlayer + 5) % 5) - 1]
        )),
        samples,
        seed: gameSeed + turn * 104729 + player * 8191,
      }
      let card
      if (player === focalPlayer) {
        const evaluations = behaviorStrategy
          ? chooseCard(observation, behaviorStrategy).evaluations
          : evaluateNeuralCards(observation, model)
        const selected = sampleEvaluation(evaluations, temperature, rng)
        card = selected.card
        trajectory.push({
          gameIndex,
          player,
          strategy: behaviorStrategy || 'neural_rl',
          turn,
          deckMode,
          rows: observation.rows,
          hand: observation.hand,
          seenCards: observation.seenCards,
          deckSize: observation.deckSize,
          playerCount: observation.playerCount,
          scores: observation.scores,
          playedCards: observation.playedCards,
          chosenCard: card,
          behaviorProbability: selected.probability,
        })
      } else {
        const offset = (player - focalPlayer + 5) % 5
        card = chooseCard(observation, opponentStrategies[offset - 1]).card
      }
      if (recordAll && player !== focalPlayer) {
        const offset = (player - focalPlayer + 5) % 5
        trajectory.push({
          gameIndex,
          player,
          strategy: opponentStrategies[offset - 1],
          turn,
          deckMode,
          rows: observation.rows,
          hand: observation.hand,
          seenCards: observation.seenCards,
          deckSize: observation.deckSize,
          playerCount: observation.playerCount,
          scores: observation.scores,
          playedCards: observation.playedCards,
          chosenCard: card,
          behaviorProbability: 1,
        })
      }
      actions.push({ playerId: player, card })
    }

    publicHistory.push(actions.map(({ card }) => card))
    actions.forEach(({ playerId, card }) => removeCard(hands[playerId], card))
    const result = resolveTurn({ rows, actions, humanPlayerId: -1 })
    if (result.needsChoice) throw new Error('竞技场训练局不应等待选行')
    rows = result.rows
    actions.forEach(({ card }) => seenCards.push(card))
    actions.forEach(({ playerId }) => { scores[playerId] += result.penalties[playerId] })
  }

  const minimum = Math.min(...scores)
  for (const record of trajectory) {
    const player = record.player ?? focalPlayer
    const finalScore = scores[player]
    const winnerCount = scores.filter((score) => score === minimum).length
    const winShare = finalScore === minimum ? 1 / winnerCount : 0
    const lower = scores.filter((score) => score < finalScore).length
    const tied = scores.filter((score) => score === finalScore).length
    const rank = 1 + lower + (tied - 1) / 2
    const reward = winShare * 5 - (rank - 1) * 0.42 - finalScore * 0.075
    await writeLine(stream, { ...record, player, reward, winShare, finalScore, finalRank: rank })
    states += 1
  }

  const reportEvery = Math.max(1, Math.floor(games / 20))
  if ((gameIndex + 1) % reportEvery === 0 || gameIndex + 1 === games) {
    console.log(`[arena-data] ${gameIndex + 1}/${games} games · ${states} states · ${((performance.now() - started) / 1000).toFixed(1)}s`)
  }
}

stream.end()
await once(stream, 'finish')
console.log(`[arena-data] wrote ${output}`)
