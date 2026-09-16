import { createWriteStream, mkdirSync, readFileSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { once } from 'node:events'
import { chooseCard, mulberry32 } from '../src/game/ai-core.js'
import { createGame, resolveTurn } from '../src/game/engine.js'
import { chooseNeuralCard } from '../src/game/neural-features.js'

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

const games = Number(argument('games', '400'))
const output = resolve(argument('output', 'artifacts/training/expert-train.jsonl'))
const baseSeed = Number(argument('seed', '20260820'))
const teacherSamples = Number(argument('samples', '40'))
const playerCounts = argument('players', '3,4,5,6').split(',').map(Number)
const deckModes = argument('modes', 'adaptive,classic').split(',').filter(Boolean)
const behaviorModelPath = argument('behavior-model', '')
const behaviorModel = behaviorModelPath ? JSON.parse(readFileSync(resolve(behaviorModelPath), 'utf8')) : null
if (!games || !teacherSamples || playerCounts.some((count) => count < 2 || count > 10)) throw new Error('训练数据参数非法')

mkdirSync(dirname(output), { recursive: true })
const stream = createWriteStream(output, { encoding: 'utf8' })
await writeLine(stream, {
  meta: {
    format: 'ntw-expert-observations-v1',
    games,
    baseSeed,
    teacherSamples,
    playerCounts,
    deckModes,
    behaviorModel: behaviorModelPath || null,
  },
})

let observations = 0
let candidates = 0
const started = performance.now()

for (let gameIndex = 0; gameIndex < games; gameIndex += 1) {
  const playerCount = playerCounts[gameIndex % playerCounts.length]
  const deckMode = deckModes[Math.floor(gameIndex / playerCounts.length) % deckModes.length]
  const gameSeed = baseSeed + gameIndex * 65537
  const behaviorRng = mulberry32(gameSeed ^ 0x74726169)
  const deal = createGame(playerCount, deckMode, mulberry32(gameSeed))
  const hands = deal.hands.map((hand) => [...hand])
  const scores = Array(playerCount).fill(0)
  let rows = deal.rows.map((row) => [...row])
  const seenCards = [...deal.initialCards]

  for (let turn = 0; turn < 10; turn += 1) {
    const actions = []
    for (let player = 0; player < playerCount; player += 1) {
      const hand = hands[player]
      const perspectiveScores = [scores[player], ...scores.filter((_, index) => index !== player)]
      const observation = {
        rows: rows.map((row) => [...row]),
        hand: [...hand],
        seenCards: [...seenCards],
        deckSize: deal.deckSize,
        playerCount,
        scores: perspectiveScores,
        samples: teacherSamples,
        seed: gameSeed + turn * 104729 + player * 8191,
      }
      const teacher = chooseCard(observation, 'champion')
      await writeLine(stream, {
        gameSeed,
        gameIndex,
        turn,
        player,
        deckMode,
        rows: observation.rows,
        hand: observation.hand,
        seenCards: observation.seenCards,
        deckSize: observation.deckSize,
        playerCount,
        scores: observation.scores,
        labels: teacher.evaluations.map(({ card, utility }) => [card, utility]),
      })
      observations += 1
      candidates += teacher.evaluations.length

      const roll = behaviorRng()
      let card = teacher.card
      if (behaviorModel) {
        if (roll >= 0.45 && roll < 0.80) card = chooseNeuralCard(observation, behaviorModel).card
        else if (roll >= 0.80 && roll < 0.88 && teacher.evaluations.length > 1) card = teacher.evaluations[1].card
        else if (roll >= 0.88 && roll < 0.96) card = chooseCard(observation, 'cautious').card
        else if (roll >= 0.96) card = chooseCard(observation, 'random').card
      } else if (roll >= 0.65 && roll < 0.77 && teacher.evaluations.length > 1) card = teacher.evaluations[1].card
      else if (roll >= 0.77 && roll < 0.93) card = chooseCard(observation, 'cautious').card
      else if (roll >= 0.93) card = chooseCard(observation, 'random').card
      actions.push({ playerId: player, card })
    }

    actions.forEach(({ card, playerId }) => removeCard(hands[playerId], card))
    const result = resolveTurn({ rows, actions, humanPlayerId: -1 })
    if (result.needsChoice) throw new Error('自动训练局不应等待选行')
    rows = result.rows
    actions.forEach(({ card }) => seenCards.push(card))
    actions.forEach(({ playerId }) => { scores[playerId] += result.penalties[playerId] })
  }

  const reportEvery = Math.max(1, Math.floor(games / 20))
  if ((gameIndex + 1) % reportEvery === 0 || gameIndex === games - 1) {
    const seconds = (performance.now() - started) / 1000
    console.log(`[data] ${gameIndex + 1}/${games} games · ${observations} states · ${candidates} candidates · ${seconds.toFixed(1)}s`)
  }
}

stream.end()
await once(stream, 'finish')
console.log(`[data] wrote ${output}`)
