import { createWriteStream, mkdirSync, readFileSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { once } from 'node:events'
import { chooseCard, mulberry32, setCounterfactualPolicyModelForEvaluation, setPolicyV2ModelForEvaluation } from '../src/game/ai-core.js'
import { createGame, resolveTurn } from '../src/game/engine.js'

function argument(name, fallback) {
  const index = process.argv.indexOf(`--${name}`)
  return index >= 0 ? process.argv[index + 1] : fallback
}

function removeCard(hand, card) {
  const index = hand.indexOf(card)
  if (index < 0) throw new Error(`无法从手牌移除 ${card}`)
  hand.splice(index, 1)
}

function cloneRows(rows) {
  return rows.map((row) => [...row])
}

function observationOf({ rows, hands, seenCards, scores, deckSize, publicHistory }, player, seed, samples = 18, strategies = null) {
  const relativePlayers = Array.from({ length: 5 }, (_, offset) => (player + offset) % 5)
  return {
    rows: cloneRows(rows),
    hand: [...hands[player]],
    seenCards: [...seenCards],
    deckSize,
    playerCount: 5,
    playerSeat: player,
    scores: relativePlayers.map((seat) => scores[seat]),
    playedCards: relativePlayers.map((seat) => (publicHistory || []).map((round) => round[seat])),
    opponentStrategies: strategies ? relativePlayers.slice(1).map((seat) => strategies[seat]) : null,
    samples,
    seed,
  }
}

function applyActions(state, cards) {
  const actions = cards.map((card, playerId) => ({ playerId, card }))
  actions.forEach(({ playerId, card }) => removeCard(state.hands[playerId], card))
  const result = resolveTurn({ rows: state.rows, actions, humanPlayerId: -1 })
  if (result.needsChoice) throw new Error('反事实 rollout 不应等待选行')
  state.rows = result.rows
  state.publicHistory.push([...cards])
  actions.forEach(({ card }) => state.seenCards.push(card))
  actions.forEach(({ playerId }) => { state.scores[playerId] += result.penalties[playerId] })
}

function rolloutCost(initialState, focalPlayer, firstCards, gameSeed, turn, strategies, realRollout, rolloutSamples) {
  const state = {
    deckSize: initialState.deckSize,
    rows: cloneRows(initialState.rows),
    hands: initialState.hands.map((hand) => [...hand]),
    seenCards: [...initialState.seenCards],
    scores: [...initialState.scores],
    publicHistory: initialState.publicHistory.map((round) => [...round]),
  }
  applyActions(state, firstCards)
  const rolloutStrategies = realRollout ? strategies : strategies.map((strategy, player) => {
    if (player === focalPlayer) return 'neural_rl'
    if (strategy === 'random') return 'random'
    if (strategy === 'external_mcs') return 'greedy'
    return 'cautious'
  })
  for (let futureTurn = turn + 1; futureTurn < 10; futureTurn += 1) {
    const cards = state.hands.map((_, player) => chooseCard(
      observationOf(state, player, gameSeed + futureTurn * 104729 + player * 8191, rolloutSamples, rolloutStrategies),
      rolloutStrategies[player],
    ).card)
    applyActions(state, cards)
  }
  const focalScore = state.scores[focalPlayer]
  const minimum = Math.min(...state.scores)
  const winnerCount = state.scores.filter((score) => score === minimum).length
  const winShare = focalScore === minimum ? 1 / winnerCount : 0
  const lower = state.scores.filter((score) => score < focalScore).length
  const tied = state.scores.filter((score) => score === focalScore).length
  const rank = 1 + lower + (tied - 1) / 2
  return focalScore * 0.12 + (rank - 1) * 0.8 - winShare * 8
}

async function writeLine(stream, value) {
  if (!stream.write(`${JSON.stringify(value)}\n`)) await once(stream, 'drain')
}

const games = Number(argument('games', '1000'))
const output = resolve(argument('output', 'artifacts/training/counterfactual.jsonl'))
const baseSeed = Number(argument('seed', '251100000'))
const deckMode = argument('mode', 'adaptive') === 'classic' ? 'classic' : 'adaptive'
const samples = Number(argument('samples', '18'))
const behaviorStrategy = argument('behavior-strategy', 'neural_hybrid_rl_k35_p0_a4_r150_v0')
const policyV2ModelPath = argument('policy-v2-model', '')
const counterfactualModelPath = argument('counterfactual-model', '')
const realRollout = argument('real-rollout', 'false') === 'true'
const rolloutSamples = Number(argument('rollout-samples', realRollout ? String(samples) : '8'))
if (policyV2ModelPath) setPolicyV2ModelForEvaluation(JSON.parse(readFileSync(resolve(policyV2ModelPath), 'utf8')))
if (counterfactualModelPath) setCounterfactualPolicyModelForEvaluation(JSON.parse(readFileSync(resolve(counterfactualModelPath), 'utf8')))
const fixedOpponents = ['champion', 'external_mcs', 'cautious', 'random']

mkdirSync(dirname(output), { recursive: true })
const stream = createWriteStream(output, { encoding: 'utf8' })
await writeLine(stream, { meta: { format: 'ntw-counterfactual-v2', games, baseSeed, deckMode, samples, behaviorStrategy, policyV2ModelPath: policyV2ModelPath || null, counterfactualModelPath: counterfactualModelPath || null, realRollout, rolloutSamples } })
const started = performance.now()
let states = 0
let candidates = 0

for (let gameIndex = 0; gameIndex < games; gameIndex += 1) {
  const gameSeed = baseSeed + gameIndex * 65537
  const focalPlayer = gameIndex % 5
  const strategies = Array(5)
  strategies[focalPlayer] = behaviorStrategy
  fixedOpponents.forEach((strategy, index) => { strategies[(focalPlayer + index + 1) % 5] = strategy })
  const deal = createGame(5, deckMode, mulberry32(gameSeed))
  const state = {
    deckSize: deal.deckSize,
    rows: cloneRows(deal.rows),
    hands: deal.hands.map((hand) => [...hand]),
    seenCards: [...deal.initialCards],
    scores: Array(5).fill(0),
    publicHistory: [],
  }

  for (let turn = 0; turn < 10; turn += 1) {
    const actualCards = state.hands.map((_, player) => chooseCard(
      observationOf(state, player, gameSeed + turn * 104729 + player * 8191, samples, strategies),
      strategies[player],
    ).card)
    const focalObservation = observationOf(state, focalPlayer, gameSeed + turn * 104729 + focalPlayer * 8191, samples, strategies)
    const labels = focalObservation.hand.map((card) => {
      const cards = [...actualCards]
      cards[focalPlayer] = card
      return [card, rolloutCost(state, focalPlayer, cards, gameSeed, turn, strategies, realRollout, rolloutSamples)]
    })
    await writeLine(stream, {
      gameIndex,
      turn,
      deckMode,
      rows: focalObservation.rows,
      hand: focalObservation.hand,
      seenCards: focalObservation.seenCards,
      deckSize: focalObservation.deckSize,
      playerCount: 5,
      playerSeat: focalObservation.playerSeat,
      scores: focalObservation.scores,
      playedCards: focalObservation.playedCards,
      labels,
    })
    states += 1
    candidates += labels.length
    applyActions(state, actualCards)
  }

  const reportEvery = Math.max(1, Math.floor(games / 20))
  if ((gameIndex + 1) % reportEvery === 0 || gameIndex + 1 === games) {
    console.log(`[counterfactual] ${gameIndex + 1}/${games} games · ${states} states · ${candidates} branches · ${((performance.now() - started) / 1000).toFixed(1)}s`)
  }
}

stream.end()
await once(stream, 'finish')
console.log(`[counterfactual] wrote ${output}`)
