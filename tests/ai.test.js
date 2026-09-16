import test from 'node:test'
import assert from 'node:assert/strict'
import { chooseCard, playArenaGame } from '../src/game/ai-core.js'

const observation = {
  rows: [[7], [18], [31], [39]],
  hand: [3, 13, 19, 24, 29, 37, 41, 45, 51, 53],
  seenCards: [7, 18, 31, 39],
  deckSize: 54,
  playerCount: 5,
  scores: [0, 0, 0, 0, 0],
  samples: 8,
  seed: 20260820,
}

test('所有竞技场策略只会选择自己的合法手牌', () => {
  for (const strategy of ['neural_hybrid', 'neural_hybrid_v2', 'policy_v2', 'neural', 'champion', 'external_mcs', 'legacy', 'cautious', 'greedy', 'random']) {
    const decision = chooseCard(observation, strategy)
    assert.ok(observation.hand.includes(decision.card), `${strategy} returned ${decision.card}`)
  }
})

test('冠军 AI 在固定信息集和种子下可复现', () => {
  const first = chooseCard(observation, 'champion')
  const second = chooseCard(observation, 'champion')
  assert.equal(first.card, second.card)
  assert.deepEqual(first.evaluations, second.evaluations)
})

test('自适应竞技场会对已公开的冠军与 MCS 使用校准后的对手模型', () => {
  const arenaObservation = {
    ...observation,
    opponentStrategies: ['champion', 'external_mcs', 'cautious', 'random'],
  }
  const released = chooseCard(arenaObservation, 'neural_hybrid')
  const calibrated = chooseCard(arenaObservation, 'neural_hybrid_zxcg')
  assert.equal(released.card, calibrated.card)
  assert.deepEqual(released.evaluations, calibrated.evaluations)
})

test('竞技场能完成十回合并返回合法分数', () => {
  const result = playArenaGame({
    playerCount: 5,
    deckMode: 'adaptive',
    strategies: ['champion', 'legacy', 'cautious', 'greedy', 'random'],
    seed: 42,
    championSamples: 8,
    legacySamples: 8,
  })
  assert.equal(result.scores.length, 5)
  result.scores.forEach((score) => assert.ok(Number.isInteger(score) && score >= 0))
})
