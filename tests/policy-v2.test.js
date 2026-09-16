import test from 'node:test'
import assert from 'node:assert/strict'
import { encodePolicyV2Action, encodePolicyV2State, validatePolicyV2 } from '../src/game/policy-v2.js'
import model from '../src/game/models/ntw-policy-v2.json' with { type: 'json' }

const observation = {
  rows: [[7, 11], [18], [31, 33, 35], [39]],
  hand: [3, 13, 19, 24, 29, 37, 41, 45],
  seenCards: [7, 11, 18, 31, 33, 35, 39, 8, 9, 12, 15, 21, 28, 42, 47, 53],
  playedCards: [[8, 28], [9, 42], [12, 47], [15, 53], [21, 33]],
  deckSize: 54,
  playerCount: 5,
  scores: [3, 0, 7, 2, 5],
}

test('Policy v2 状态与动作特征契约固定', () => {
  assert.equal(encodePolicyV2State(observation).length, 420)
  observation.hand.forEach((card) => assert.equal(encodePolicyV2Action(observation, card).length, 17))
  assert.equal(validatePolicyV2(model), true)
})

test('Policy v2 不编码隐藏对手手牌', () => {
  const first = encodePolicyV2State({ ...observation, opponentHands: [[1, 2], [4, 5]] })
  const second = encodePolicyV2State({ ...observation, opponentHands: [[50, 51], [52, 54]] })
  assert.deepEqual(first, second)
})
