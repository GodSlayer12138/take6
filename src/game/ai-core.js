import {
  bullHeads,
  cheapestRowIndex,
  createGame,
  rowPenalty,
  targetRowIndex,
} from './engine.js'
import { chooseExternalMcsCard } from './external-adapter.js'
import { chooseNeuralCard, evaluateNeuralCards } from './neural-features.js'
import { chooseCompactOpponentCard, chooseCompactPolicyV2Card, choosePolicyV2Card, compactOpponentLogProbability, encodePolicyV2Action, encodePolicyV2State, evaluateCompactOpponentCards, evaluateCompactPolicyV2Cards, evaluatePolicyV2Cards, forwardLayers } from './policy-v2.js'
import neuralModel from './models/ntw-champion.json' with { type: 'json' }
import classicFiveModel from './models/ntw-classic5.json' with { type: 'json' }
import adaptiveOutcomeModel from './models/ntw-adaptive-rl.json' with { type: 'json' }
import classicOutcomeModel from './models/ntw-classic5-rl.json' with { type: 'json' }
import adaptiveFiveModel from './models/ntw-adaptive5.json' with { type: 'json' }
import defaultPolicyV2Model from '../../artifacts/models/ntw-policy-v2-distill2-t120-i050.json' with { type: 'json' }
import compactChampionModel from '../../artifacts/models/ntw-compact-champion.json' with { type: 'json' }
import compactExternalModel from '../../artifacts/models/ntw-compact-external_mcs.json' with { type: 'json' }
import contextChampionModel from '../../artifacts/models/ntw-context-champion.json' with { type: 'json' }
import contextExternalModel from '../../artifacts/models/ntw-context-external_mcs.json' with { type: 'json' }
import contextChampionModelV2 from '../../artifacts/models/ntw-context2-champion.json' with { type: 'json' }
import contextExternalModelV2 from '../../artifacts/models/ntw-context2-external.json' with { type: 'json' }
import compactChampionModelV2 from '../../artifacts/models/ntw-compact2-champion.json' with { type: 'json' }
import compactExternalModelV2 from '../../artifacts/models/ntw-compact2-external_mcs.json' with { type: 'json' }
import distilledTeacherPolicyV2Model from '../../artifacts/models/ntw-policy-v2-distill2-t120.json' with { type: 'json' }
import realArenaAwrPolicyV2Model from '../../artifacts/models/ntw-policy-v2-kr10k-avg-a10-b05.json' with { type: 'json' }
import compactFocalPolicyModel from '../../artifacts/models/ntw-compact-focal-v1.json' with { type: 'json' }
import compactFocalPolicyT05Model from '../../artifacts/models/ntw-compact-focal-t05-v1.json' with { type: 'json' }
import compactCounterfactualPolicyModel from '../../artifacts/models/ntw-compact-counterfactual-final.json' with { type: 'json' }
import compactCounterfactualPolicyT05Model from '../../artifacts/models/ntw-compact-counterfactual-t05-v1.json' with { type: 'json' }
import compactFullHistoryPolicyModel from '../../artifacts/models/ntw-policy-v2-compact-fullhist-v1.json' with { type: 'json' }
import realArenaValuePolicyV2Model from '../../artifacts/models/ntw-value-v2-real4.json' with { type: 'json' }
import realArenaKrValuePolicyV2Model from '../../artifacts/models/ntw-value-v2-kr10k-cost.json' with { type: 'json' }
import realArenaWinValuePolicyV2Model from '../../artifacts/models/ntw-winvalue-v2-real4.json' with { type: 'json' }
import realArenaWinValueNewPolicyV2Model from '../../artifacts/models/ntw-winvalue-v2-kr-new.json' with { type: 'json' }
import realArenaWinValueMixedPolicyV2Model from '../../artifacts/models/ntw-winvalue-v2-kr-mixed.json' with { type: 'json' }
import handPosteriorModel from '../../artifacts/models/ntw-hand-posterior-v1.json' with { type: 'json' }
import portfolioGateModel from '../../artifacts/models/ntw-portfolio-gate-v1.json' with { type: 'json' }
import searchCounterfactualGateModel from '../../artifacts/models/ntw-search-gate-v1.json' with { type: 'json' }
import searchResidualModel from '../../artifacts/models/ntw-search-residual-v2.json' with { type: 'json' }

let policyV2Model = defaultPolicyV2Model
let counterfactualRolloutModel = compactCounterfactualPolicyModel
let counterfactualRootModel = compactCounterfactualPolicyModel

export function setPolicyV2ModelForEvaluation(model) {
  policyV2Model = model || defaultPolicyV2Model
}

export function setCounterfactualPolicyModelForEvaluation(model) {
  counterfactualRolloutModel = model || compactCounterfactualPolicyModel
  counterfactualRootModel = model || compactCounterfactualPolicyModel
}

export function setCounterfactualPolicyModelsForEvaluation({ root = null, rollout = null } = {}) {
  counterfactualRootModel = root || compactCounterfactualPolicyModel
  counterfactualRolloutModel = rollout || compactCounterfactualPolicyModel
}

function evaluateCounterfactualCards(observation, model = counterfactualRootModel) {
  if (model?.format === 'ntw-counterfactual-gated') {
    const [newModel, oldModel] = model.models || []
    if (!newModel || !oldModel || !model.gateLayers?.length) throw new Error('反事实门控模型不完整')
    const newEntries = evaluateCounterfactualCards(observation, newModel)
    const oldEntries = evaluateCounterfactualCards(observation, oldModel)
    if (observation.hand.length <= 1) return newEntries
    const newCard = newEntries[0].card
    const oldCard = oldEntries[0].card
    const newMargin = (newEntries[0].logit - newEntries[1].logit) / 5
    const oldMargin = (oldEntries[0].logit - oldEntries[1].logit) / 5
    const gateFeatures = [
      ...encodePolicyV2State(observation),
      newCard / 54,
      oldCard / 54,
      Math.max(-2, Math.min(2, newMargin)),
      Math.max(-2, Math.min(2, oldMargin)),
      Number(newCard === oldCard),
      (newCard - oldCard) / 54,
    ]
    return forwardLayers(gateFeatures, model.gateLayers)[0] > 0 ? newEntries : oldEntries
  }
  if (model?.format === 'ntw-counterfactual-ensemble') {
    const members = model.models || []
    if (!members.length) throw new Error('反事实集成模型缺少成员')
    const configured = model.weights?.length === members.length ? model.weights : members.map(() => 1)
    const totalWeight = configured.reduce((sum, value) => sum + Math.max(0, value), 0) || members.length
    const ranked = members.map((member) => evaluateCounterfactualCards(observation, member))
    const rankMaps = ranked.map((entries) => new Map(entries.map((entry, index) => [entry.card, index])))
    return observation.hand.map((card) => ({
      card,
      utility: rankMaps.reduce((sum, ranks, index) => sum + ranks.get(card) * Math.max(0, configured[index]) / totalWeight, 0),
      risk: 0,
      disasterRate: 0,
    })).sort((left, right) => left.utility - right.utility || left.card - right.card)
  }
  if (model?.format === 'ntw-policy-v2') {
    return evaluatePolicyV2Cards(observation, model)
  }
  if (model?.format === 'ntw-policy-v2-compact') {
    return evaluateCompactPolicyV2Cards(observation, model)
  }
  return evaluateCompactOpponentCards(observation, model).map((entry) => ({
    ...entry,
    utility: -entry.logit,
    risk: 0,
    disasterRate: 0,
  }))
}

function choosePortfolioStrategy(observation) {
  if (portfolioGateModel?.format !== 'ntw-portfolio-gate-v1' || !portfolioGateModel.layers?.length) {
    throw new Error('开局策略门控模型不完整')
  }
  const logits = forwardLayers(encodePolicyV2State(observation), portfolioGateModel.layers)
  let selected = 0
  for (let index = 1; index < logits.length; index += 1) {
    if (logits[index] > logits[selected]) selected = index
  }
  return portfolioGateModel.strategies[selected]
}

function chooseCounterfactualCard(observation) {
  return evaluateCounterfactualCards(observation, counterfactualRolloutModel)[0].card
}

function applySearchResidual(observation, searchResult, blend) {
  const entries = searchResult.evaluations
  if (entries.length <= 1) return searchResult
  const mean = (key, fallback = 0) => entries.reduce((sum, entry) => sum + Number(entry[key] ?? fallback), 0) / entries.length
  const meanUtility = mean('utility')
  const meanSearch = entries.reduce((sum, entry) => sum + Number(entry.searchUtility ?? entry.utility), 0) / entries.length
  const meanNeural = mean('neuralUtility')
  const clip = (value) => Math.max(-3, Math.min(3, value))
  const context = forwardLayers(encodePolicyV2State(observation), searchResidualModel.stateLayers, true)
  const corrected = entries.map((entry, rank) => {
    const dynamic = [
      clip((entry.utility - meanUtility) / 5),
      clip(((entry.searchUtility ?? entry.utility) - meanSearch) / 5),
      clip(((entry.neuralUtility ?? 0) - meanNeural) / 5),
      Math.min(2, (entry.expectedPenalty ?? 0) / 15),
      Math.min(2, (entry.immediatePenalty ?? 0) / 15),
      Math.min(3, (entry.risk ?? 0) / 10),
      clip(((entry.cvar ?? 0) - (entry.expectedPenalty ?? 0)) / 10),
      Math.max(0, Math.min(1, entry.disasterRate ?? 0)),
      rank / Math.max(1, entries.length - 1),
      Math.min(2, (entry.samples ?? searchResult.sampleCount ?? 18) / 18),
    ]
    const delta = forwardLayers([...context, ...encodePolicyV2Action(observation, entry.card), ...dynamic], searchResidualModel.deltaLayers)[0]
    return { ...entry, residualDelta: delta, correctedLogit: -(entry.utility - meanUtility) / 5 + delta * blend }
  }).sort((left, right) => right.correctedLogit - left.correctedLogit || left.card - right.card)
  return { ...searchResult, card: corrected[0].card, residualCorrection: true, residualEvaluations: corrected }
}

export const STRATEGIES = {
  neural_hybrid: { id: 'neural_hybrid', name: '牧场终极冠军', shortName: '终极冠军', description: '4060 训练的神经先验、动态候选与反事实续局搜索', color: '#f2c14e' },
  neural_hybrid_rl: { id: 'neural_hybrid_rl', name: '终局强化候选', shortName: '强化候选', description: '使用完整对局胜负奖励微调的候选策略', color: '#ffd166' },
  neural_hybrid_ensemble: { id: 'neural_hybrid_ensemble', name: '双模型共识候选', shortName: '共识候选', description: '蒸馏模型与终局强化模型按候选排名集成', color: '#f7d774' },
  neural_hybrid_spec: { id: 'neural_hybrid_spec', name: '五人专用候选', shortName: '五人专用', description: '只使用五人自适应牌堆教师数据训练', color: '#e9c46a' },
  neural_hybrid_v2: { id: 'neural_hybrid_v2', name: '历史 Value 候选', shortName: 'Policy v2', description: 'GPU 向量化 PPO、逐座位公共历史与终局 value 搜索先验', color: '#65d6c4' },
  policy_v2: { id: 'policy_v2', name: '纯 Policy v2', shortName: '纯 v2', description: '不使用搜索的 GPU PPO 策略', color: '#55b9a9' },
  neural: { id: 'neural', name: 'GPU 神经策略', shortName: '纯神经', description: '蒸馏高预算搜索的 270 维动作价值网络', color: '#e96b7b' },
  neural_rl: { id: 'neural_rl', name: '终局强化纯网络', shortName: '强化纯网络', description: '不使用搜索，隔离评估终局强化网络', color: '#ff8c69' },
  champion: { id: 'champion', name: '牧场冠军 v2', shortName: '冠军 v2', description: '共享信息集、对手建模、尾部风险与整局推演', color: '#ef9c5d' },
  external_mcs: { id: 'external_mcs', name: '公开 MCS · 适配', shortName: '外部 MCS', description: '移植 rl-6-nimmt 的随机整局蒙特卡洛搜索', color: '#d8b15f' },
  legacy: { id: 'legacy', name: '蒙特卡洛 v1', shortName: '搜索 v1', description: '上一版独立采样的整局 rollout', color: '#7eb89d' },
  cautious: { id: 'cautious', name: '风险规避', shortName: '风险规避', description: '计算牌列拥堵概率与未来手牌形状', color: '#75a9d1' },
  greedy: { id: 'greedy', name: '即时贪心', shortName: '即时贪心', description: '只选择当前看起来损失最小的牌', color: '#b18ad1' },
  random: { id: 'random', name: '随机基线', shortName: '随机', description: '从合法手牌中完全随机选择', color: '#8b9a95' },
}

export const FINAL_ADAPTIVE_STRATEGY = 'neural_hybrid_v2_q4443333_crn_ex3_cp5_cf_bm8_p0_ls_e50_u_bh4_hc_bt5_kr_iw15_zxtt_kq_fq3_kx10'

export function mulberry32(seed) {
  let state = seed >>> 0
  return () => {
    state += 0x6d2b79f5
    let value = state
    value = Math.imul(value ^ (value >>> 15), value | 1)
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61)
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296
  }
}

function neuralModelFor(observation) {
  const classicMode = observation.deckSize > observation.playerCount * 10 + 4
  return classicMode && observation.playerCount === 5 ? classicFiveModel : neuralModel
}

function outcomeModelFor(observation) {
  const classicMode = observation.deckSize > observation.playerCount * 10 + 4
  return classicMode ? classicOutcomeModel : adaptiveOutcomeModel
}

function shuffleInPlace(items, rng) {
  for (let index = items.length - 1; index > 0; index -= 1) {
    const swapIndex = Math.floor(rng() * (index + 1))
    ;[items[index], items[swapIndex]] = [items[swapIndex], items[index]]
  }
  return items
}

function radicalInverse(index, base) {
  let value = 0
  let factor = 1 / base
  let remaining = index
  while (remaining > 0) {
    value += (remaining % base) * factor
    remaining = Math.floor(remaining / base)
    factor /= base
  }
  return value
}

function cloneRows(rows) {
  return rows.map((row) => [...row])
}

function removeCard(hand, card) {
  const index = hand.indexOf(card)
  if (index >= 0) hand.splice(index, 1)
}

function playCards(rows, cards) {
  const nextRows = cloneRows(rows)
  const penalties = Array(cards.length).fill(0)
  const ordered = cards.map((card, player) => ({ card, player })).sort((left, right) => left.card - right.card)

  for (const { card, player } of ordered) {
    let rowIndex = targetRowIndex(nextRows, card)
    const tooLow = rowIndex < 0
    if (tooLow) rowIndex = cheapestRowIndex(nextRows)
    if (tooLow || nextRows[rowIndex].length >= 5) {
      penalties[player] += rowPenalty(nextRows[rowIndex])
      nextRows[rowIndex] = [card]
    } else {
      nextRows[rowIndex].push(card)
    }
  }
  return { rows: nextRows, penalties }
}

function unseenCards(observation) {
  const known = new Set([...observation.seenCards, ...observation.hand])
  const unseen = []
  for (let card = 1; card <= observation.deckSize; card += 1) {
    if (!known.has(card)) unseen.push(card)
  }
  return unseen
}

function placementInfo(card, rows) {
  const rowIndex = targetRowIndex(rows, card)
  if (rowIndex < 0) {
    const forcedRow = cheapestRowIndex(rows)
    return { rowIndex: forcedRow, immediate: rowPenalty(rows[forcedRow]), gap: 0, openSlots: 0, tooLow: true }
  }
  const row = rows[rowIndex]
  return {
    rowIndex,
    immediate: row.length >= 5 ? rowPenalty(row) : 0,
    gap: card - row[row.length - 1] - 1,
    openSlots: 5 - row.length,
    tooLow: false,
  }
}

function poissonTail(lambda, threshold) {
  if (threshold <= 0) return 1
  if (lambda <= 0) return 0
  let term = Math.exp(-lambda)
  let cumulative = term
  for (let value = 1; value < threshold; value += 1) {
    term *= lambda / value
    cumulative += term
  }
  return Math.max(0, Math.min(1, 1 - cumulative))
}

function strategicCost(card, hand, rows, observation) {
  const info = placementInfo(card, rows)
  if (info.tooLow || info.immediate > 0) {
    const escapeBonus = info.tooLow && info.immediate <= 2 ? 0.45 : 0
    return info.immediate * 4.8 - escapeBonus
  }

  const row = rows[info.rowIndex]
  const unknown = observation._unseen || unseenCards(observation)
  const tail = row[row.length - 1]
  let intervalCount = 0
  for (const unknownCard of unknown) {
    if (unknownCard > tail && unknownCard < card) intervalCount += 1
  }
  const expectedTraffic = (intervalCount / Math.max(1, unknown.length)) * (observation.playerCount - 1)
  const overflowChance = poissonTail(expectedTraffic, Math.max(1, info.openSlots))
  const congestion = overflowChance * (0.8 + rowPenalty(row) * 0.62)
  const distance = Math.min(30, info.gap) * 0.014
  const remaining = hand.filter((value) => value !== card)
  const minTail = Math.min(...rows.map((candidateRow) => candidateRow[candidateRow.length - 1]))
  const trapped = remaining.filter((value) => value < minTail).length
  const adjacency = remaining.some((value) => value > card && value - card <= 2) ? -0.14 : 0
  const shedExpensiveCard = -(bullHeads(card) - 1) * 0.045
  const endgameWeight = 1 + Math.max(0, 5 - hand.length) * 0.08
  return (congestion + distance + trapped * 0.11 + adjacency + shedExpensiveCard) * endgameWeight
}

function rankedChoices(hand, rows, observation) {
  const prepared = { ...observation, _unseen: observation._unseen || unseenCards(observation) }
  return hand
    .map((card) => ({ card, cost: strategicCost(card, hand, rows, prepared) }))
    .sort((left, right) => left.cost - right.cost || left.card - right.card)
}

function greedyChoice(hand, rows) {
  return [...hand]
    .map((card) => {
      const info = placementInfo(card, rows)
      return { card, cost: info.immediate * 100 + info.gap + (info.openSlots <= 1 ? 8 : 0) }
    })
    .sort((left, right) => left.cost - right.cost || left.card - right.card)[0].card
}

function cautiousChoice(hand, rows, observation, rng = null) {
  const ranked = rankedChoices(hand, rows, observation)
  if (rng && ranked.length > 1 && rng() < 0.11) return ranked[1].card
  return ranked[0].card
}

function tempoChoice(hand, rows, observation, rng) {
  const ranked = rankedChoices(hand, rows, observation).slice(0, Math.min(4, hand.length))
  const progress = 1 - hand.length / 10
  const target = observation.deckSize * (progress < 0.45 ? 0.28 : 0.62)
  ranked.sort((left, right) => (left.cost * 7 + Math.abs(left.card - target) / observation.deckSize) - (right.cost * 7 + Math.abs(right.card - target) / observation.deckSize))
  if (rng && ranked.length > 1 && rng() < 0.13) return ranked[1].card
  return ranked[0].card
}

function policyChoice(style, hand, rows, observation, rng) {
  if (style === 'random') return hand[Math.floor(rng() * hand.length)]
  if (style === 'greedy') return greedyChoice(hand, rows)
  if (style === 'tempo') return tempoChoice(hand, rows, observation, rng)
  return cautiousChoice(hand, rows, observation, rng)
}

function sampleInformationSet(observation, rng) {
  const unknown = unseenCards(observation)
  shuffleInPlace(unknown, rng)
  const hands = []
  let cursor = 0
  for (let player = 1; player < observation.playerCount; player += 1) {
    hands.push(unknown.slice(cursor, cursor + observation.hand.length))
    cursor += observation.hand.length
  }
  return hands
}

function sampleInformationSetStratified(observation, baseSeed, sampleIndex, balancedPhases = false, rotatePhases = false) {
  const opponentCount = observation.playerCount - 1
  const group = Math.floor(sampleIndex / opponentCount)
  const rawPhase = sampleIndex % opponentCount
  const phaseOrder = opponentCount === 4 ? [0, 2, 1, 3] : Array.from({ length: opponentCount }, (_, index) => index)
  const basePhase = balancedPhases ? phaseOrder[rawPhase] : rawPhase
  const phaseOffset = rotatePhases
    ? Math.floor(mulberry32(baseSeed ^ 0x53545241)() * opponentCount)
    : 0
  const phase = (basePhase + phaseOffset) % opponentCount
  const unknown = unseenCards(observation)
  const groupRng = mulberry32(baseSeed ^ Math.imul(group + 1, 0x6d2b79f5))
  shuffleInPlace(unknown, groupRng)
  const hands = Array.from({ length: opponentCount }, () => [])
  for (let row = 0; row < observation.hand.length; row += 1) {
    // Within a group, every unknown card visits every opponent exactly once.
    // Alternating coprime rotations avoids preserving the same card pairs.
    const stride = row % 2 === 0 ? 1 : opponentCount - 1
    const shift = phase * stride % opponentCount
    for (let column = 0; column < opponentCount; column += 1) {
      hands[(column + shift) % opponentCount].push(unknown[row * opponentCount + column])
    }
  }
  hands.forEach((hand) => hand.sort((left, right) => left - right))
  return hands
}

function sampleInformationSetPosterior(observation, rng) {
  const unknown = unseenCards(observation)
  const opponentCount = observation.playerCount - 1
  const handSize = observation.hand.length
  if (observation.deckSize !== 54 || opponentCount !== 4 || unknown.length !== opponentCount * handSize) {
    return sampleInformationSet(observation, rng)
  }
  const logits = forwardLayers(encodePolicyV2State(observation), handPosteriorModel.layers)
  const orderedCards = shuffleInPlace([...unknown], rng)
  const hands = Array.from({ length: opponentCount }, () => [])
  const capacities = Array(opponentCount).fill(handSize)
  for (const card of orderedCards) {
    const cardLogits = Array.from({ length: opponentCount }, (_, opponent) => logits[(card - 1) * opponentCount + opponent])
    const maximum = Math.max(...cardLogits.filter((_value, opponent) => capacities[opponent] > 0))
    const weights = cardLogits.map((value, opponent) => capacities[opponent] > 0 ? Math.exp(value - maximum) : 0)
    let roll = rng() * weights.reduce((sum, value) => sum + value, 0)
    let selected = weights.findIndex((weight) => {
      roll -= weight
      return roll <= 0 && weight > 0
    })
    if (selected < 0) selected = capacities.findIndex((capacity) => capacity > 0)
    hands[selected].push(card)
    capacities[selected] -= 1
  }
  hands.forEach((hand) => hand.sort((left, right) => left - right))
  return hands
}

function useHandPosteriorSample(options, sampleIndex) {
  if (!options.handPosterior) return false
  const fraction = options.handPosteriorFraction ?? 1
  if (fraction >= 1) return true
  if (fraction <= 0) return false
  // Keep the posterior-guided samples spread across the whole Monte Carlo
  // sequence instead of consuming one contiguous block of the budget.
  return radicalInverse(sampleIndex + 1, 2) < fraction
}

function sampleOpponentStyle(rng, profile = 'mixed') {
  const roll = rng()
  if (profile === 'arena1') return roll < 0.75 ? 'cautious' : 'random'
  if (profile === 'arena2') return roll < 0.50 ? 'cautious' : roll < 0.75 ? 'greedy' : 'random'
  if (profile === 'arena3') return roll < 0.25 ? 'cautious' : roll < 0.60 ? 'greedy' : roll < 0.75 ? 'tempo' : 'random'
  if (profile === 'arena4') return roll < 0.25 ? 'cautious' : roll < 0.50 ? 'greedy' : roll < 0.75 ? 'tempo' : 'random'
  if (profile === 'arena5') return roll < 0.50 ? 'greedy' : roll < 0.75 ? 'tempo' : 'random'
  if (roll < 0.45) return 'cautious'
  if (roll < 0.70) return 'greedy'
  if (roll < 0.90) return 'tempo'
  return 'random'
}

function historyLikelihood(observation, opponentHands, styles, mode = 'exact') {
  if (!observation.history?.length) return 0
  let score = 0
  observation.history.forEach((round, roundIndex) => {
    opponentHands.forEach((remainingHand, opponentIndex) => {
      const style = styles[opponentIndex]
      if (style === 'random') return
      if (mode === 'cautious' && observation.opponentStrategies?.[opponentIndex] !== 'cautious') return
      if (mode === 'seededCautious' && observation.opponentStrategies?.[opponentIndex] !== 'cautious') return
      const laterCards = observation.history.slice(roundIndex).map((entry) => entry.opponentCards[opponentIndex])
      const reconstructed = [...remainingHand, ...laterCards].sort((left, right) => left - right)
      const observed = round.opponentCards[opponentIndex]
      const likelihoodStyle = style === 'tempo' ? 'cautious' : style
      const likelihoodObservation = {
        ...observation,
        rows: round.rows,
        hand: reconstructed,
        seenCards: round.seenCards,
      }
      if (mode === 'seededCautious') {
        // The arena cautious policy's exploration bit is a deterministic
        // function of the public game/turn/seat seed.  Conditioning sampled
        // hands on the exact historical action is therefore stronger and
        // less biased than merely preferring a low heuristic rank.
        const turnOffset = roundIndex - observation.history.length
        const predicted = seededCautiousChoice(
          observation,
          reconstructed,
          opponentIndex + 1,
          0.11,
          turnOffset,
          round.rows,
          round.seenCards,
          observation.scores,
        )
        score += predicted === observed ? 1.5 : -2.5
      } else if (mode === 'rank' || mode === 'cautious') {
        const ordered = likelihoodStyle === 'greedy'
          ? [...reconstructed].sort((left, right) => {
              const leftInfo = placementInfo(left, round.rows)
              const rightInfo = placementInfo(right, round.rows)
              const leftCost = leftInfo.immediate * 100 + leftInfo.gap + (leftInfo.openSlots <= 1 ? 8 : 0)
              const rightCost = rightInfo.immediate * 100 + rightInfo.gap + (rightInfo.openSlots <= 1 ? 8 : 0)
              return leftCost - rightCost || left - right
            })
          : rankedChoices(reconstructed, round.rows, likelihoodObservation).map((entry) => entry.card)
        const rank = Math.max(0, ordered.indexOf(observed))
        score -= rank / Math.max(1, ordered.length - 1)
      } else {
        const predicted = policyChoice(likelihoodStyle, reconstructed, round.rows, likelihoodObservation, null)
        score += predicted === observed ? 1.5 : -0.35
      }
    })
  })
  return score
}

function simulationPerspective(observation, hand, player, seed) {
  const players = Array.from({ length: observation.playerCount }, (_, offset) => (player + offset) % observation.playerCount)
  const scores = observation.scores || Array(observation.playerCount).fill(0)
  const playedCards = observation.playedCards || Array.from({ length: observation.playerCount }, () => [])
  return {
    ...observation,
    hand,
    playerSeat: Number.isInteger(observation.playerSeat)
      ? (observation.playerSeat + player) % observation.playerCount
      : observation.playerSeat,
    scores: players.map((seat) => scores[seat]),
    playedCards: players.map((seat) => playedCards[seat] || []),
    opponentStrategies: undefined,
    samples: Math.max(1, observation._opponentPredictSamples || 1),
    seed,
  }
}

function opponentProxyModel(strategy, contextual, version2 = false) {
  if (strategy === 'champion') {
    if (contextual === 2) return contextChampionModelV2
    if (contextual) return contextChampionModel
    return version2 ? compactChampionModelV2 : compactChampionModel
  }
  if (contextual === 2) return contextExternalModelV2
  if (contextual) return contextExternalModel
  return version2 ? compactExternalModelV2 : compactExternalModel
}

function proxyHistoryLikelihood(observation, opponentHands, temperature, contextual = false) {
  if (!observation.history?.length) return 0
  let score = 0
  observation.history.forEach((round, roundIndex) => {
    opponentHands.forEach((remainingHand, opponentIndex) => {
      const strategy = observation.opponentStrategies?.[opponentIndex]
      if (strategy !== 'champion' && strategy !== 'external_mcs') return
      const laterCards = observation.history.slice(roundIndex).map((entry) => entry.opponentCards[opponentIndex])
      const reconstructed = [...remainingHand, ...laterCards].sort((left, right) => left - right)
      const likelihoodObservation = { ...observation, rows: round.rows, hand: reconstructed, seenCards: round.seenCards }
      score += compactOpponentLogProbability(likelihoodObservation, opponentProxyModel(strategy, contextual), round.opponentCards[opponentIndex], temperature)
    })
  })
  return score
}

function knownOpponentSeed(observation, relativePlayer, turnOffset = 0) {
  if (!Number.isInteger(observation.playerSeat)) return null
  const focalSeat = observation.playerSeat
  const opponentSeat = (focalSeat + relativePlayer) % observation.playerCount
  return observation.seed + turnOffset * 104729 + (opponentSeat - focalSeat) * 8191
}

function knownRandomChoice(observation, hand, relativePlayer, turnOffset = 0) {
  const opponentSeed = knownOpponentSeed(observation, relativePlayer, turnOffset)
  if (opponentSeed == null) return null
  const orderedHand = [...hand].sort((left, right) => left - right)
  return orderedHand[Math.floor(mulberry32(opponentSeed)() * orderedHand.length)]
}

function seededCautiousChoice(observation, hand, relativePlayer, exploration, turnOffset = 0, rows = observation.rows, seenCards = observation.seenCards, scores = observation.scores) {
  const opponentSeed = knownOpponentSeed(observation, relativePlayer, turnOffset)
  if (opponentSeed == null) return null
  const ranked = rankedChoices(hand, rows, { ...observation, rows, hand, seenCards, scores })
  return ranked.length > 1 && mulberry32(opponentSeed)() < exploration ? ranked[1].card : ranked[0].card
}

function predictKnownOpponentAction(observation, hand, opponentIndex, style, rng, sampleSeed, predictSamples, compactTemperature = null, compactContextual = false, compactTarget = 'both', compactVersion2 = false, deterministicKnownCautious = false, deterministicKnownRandom = false, compactActionQuantile = null, seededCautiousExploration = null, opponentPredictTarget = 'both') {
  const strategy = observation.opponentStrategies?.[opponentIndex]
  if (seededCautiousExploration != null && strategy === 'cautious') {
    const seeded = seededCautiousChoice(observation, hand, opponentIndex + 1, seededCautiousExploration)
    if (seeded != null) return seeded
  }
  if (deterministicKnownCautious && strategy === 'cautious') {
    return cautiousChoice(hand, observation.rows, { ...observation, hand }, null)
  }
  if (deterministicKnownRandom && strategy === 'random') {
    const exact = knownRandomChoice(observation, hand, opponentIndex + 1)
    if (exact != null) return exact
  }
  const targetMatches = compactTarget === 'both' || compactTarget === strategy
  const predictTargetMatches = opponentPredictTarget === 'both' || opponentPredictTarget === strategy
  const selectedTemperature = compactTemperature && typeof compactTemperature === 'object' ? compactTemperature[strategy] : compactTemperature
  if ((!predictSamples || !predictTargetMatches) && selectedTemperature != null && targetMatches && (strategy === 'champion' || strategy === 'external_mcs')) {
    const player = opponentIndex + 1
    const proxyObservation = simulationPerspective(observation, hand, player, sampleSeed)
    return chooseCompactOpponentCard(proxyObservation, opponentProxyModel(strategy, compactContextual, compactVersion2), compactActionQuantile == null ? rng : () => compactActionQuantile, selectedTemperature)
  }
  if (!predictSamples || !strategy || !predictTargetMatches) return policyChoice(style, hand, observation.rows, { ...observation, hand }, rng)
  const player = opponentIndex + 1
  // The arena seed is public and is derived only from the deal, turn, and
  // absolute seat.  Reconstruct the opponent's real per-turn seed instead of
  // injecting the focal search's determinization seed into an otherwise
  // deterministic opponent.  The sampled hand remains the only hidden-state
  // hypothesis used here.
  const opponentSeed = knownOpponentSeed(observation, player)
  const predictedObservation = simulationPerspective(
    { ...observation, _opponentPredictSamples: predictSamples },
    hand,
    player,
    opponentSeed ?? (sampleSeed ^ Math.imul(player + 1, 0x45d9f3b)),
  )
  if (strategy === 'champion') {
    return searchChoice(predictedObservation, true, { opponentProfile: 'mixed' }).card
  }
  if (strategy === 'external_mcs') {
    return chooseExternalMcsCard(predictedObservation, { mcMax: Math.max(10, predictSamples * 10) }).card
  }
  if (strategy === 'greedy' || strategy === 'cautious' || strategy === 'random') {
    return policyChoice(strategy, hand, observation.rows, predictedObservation, rng)
  }
  return policyChoice(style, hand, observation.rows, predictedObservation, rng)
}

function createDeterminization(observation, sampleSeed, legacy = false, opponentProfile = 'mixed', historyAttempts = 0, historyMode = 'exact', opponentPredictSamples = 0, compactOpponentTemperature = null, compactOpponentFuture = true, compactOpponentContextual = false, compactHistoryAttempts = 0, compactOpponentTarget = 'both', compactOpponentVersion2 = false, stratifiedSampleIndex = -1, stratifiedBaseSeed = 0, deterministicKnownCautious = false, deterministicKnownRandom = false, deterministicKnownCautiousRoot = false, compactStratifiedActions = false, deterministicTempo = false, seededCautiousExploration = null, balancedStratified = false, rotateStratifiedPhases = false, useHandPosterior = false, opponentPredictTarget = 'both') {
  const rng = mulberry32(sampleSeed)
  const exactProfile = opponentProfile.match(/^known-([cgtr])-([cgtr])$/)
  const decodeStyle = (code) => ({ c: 'cautious', g: 'greedy', t: 'tempo', r: 'random' })[code]
  const knownStyles = observation.opponentStrategies?.map((strategy) => {
    if (strategy === 'random') return 'random'
    if (strategy === 'cautious') return 'cautious'
    if (exactProfile && strategy === 'champion') return decodeStyle(exactProfile[1])
    if (exactProfile && strategy === 'external_mcs') return decodeStyle(exactProfile[2])
    if (opponentProfile === 'known2') return strategy === 'champion' ? 'tempo' : 'greedy'
    return 'cautious'
  })
  const styles = Array.from({ length: observation.playerCount - 1 }, (_, index) => (
    opponentProfile.startsWith('known') && knownStyles?.[index]
      ? knownStyles[index]
      : opponentProfile === 'random'
        ? 'random'
        : legacy
          ? (rng() < 0.75 ? 'cautious' : 'random')
          : sampleOpponentStyle(rng, opponentProfile)
  ))
  let opponentHands = useHandPosterior
    ? sampleInformationSetPosterior(observation, rng)
    : stratifiedSampleIndex >= 0
      ? sampleInformationSetStratified(observation, stratifiedBaseSeed, stratifiedSampleIndex, balancedStratified, rotateStratifiedPhases)
      : sampleInformationSet(observation, rng)
  if (compactHistoryAttempts > 1 && typeof compactOpponentTemperature === 'number' && observation.history?.length) {
    const jointScore = (hands) => proxyHistoryLikelihood(observation, hands, compactOpponentTemperature, compactOpponentContextual)
      + (historyAttempts > 1 ? historyLikelihood(observation, hands, styles, historyMode) : 0)
    let bestScore = jointScore(opponentHands)
    const attempts = Math.max(compactHistoryAttempts, historyAttempts)
    for (let attempt = 1; attempt < attempts; attempt += 1) {
      const candidateHands = sampleInformationSet(observation, rng)
      const candidateScore = jointScore(candidateHands)
      if (candidateScore > bestScore) {
        opponentHands = candidateHands
        bestScore = candidateScore
      }
    }
  } else if (historyAttempts > 1 && observation.history?.length) {
    let bestScore = historyLikelihood(observation, opponentHands, styles, historyMode)
    for (let attempt = 1; attempt < historyAttempts; attempt += 1) {
      const candidateHands = sampleInformationSet(observation, rng)
      const candidateScore = historyLikelihood(observation, candidateHands, styles, historyMode)
      if (candidateScore > bestScore) {
        opponentHands = candidateHands
        bestScore = candidateScore
      }
    }
  }
  const firstOpponentActions = opponentHands.map((hand, index) => {
    let compactActionQuantile = null
    if (compactStratifiedActions && stratifiedSampleIndex >= 0) {
      const bases = [2, 3, 5, 7]
      const shift = mulberry32(stratifiedBaseSeed ^ Math.imul(index + 1, 0x45d9f3b))()
      compactActionQuantile = (radicalInverse(stratifiedSampleIndex + 1, bases[index]) + shift) % 1
    }
    return predictKnownOpponentAction(
      observation,
      hand,
      index,
      styles[index],
      rng,
      sampleSeed,
      opponentPredictSamples,
      compactOpponentTemperature,
      compactOpponentContextual,
      compactOpponentTarget,
      compactOpponentVersion2,
      deterministicKnownCautious || deterministicKnownCautiousRoot,
      deterministicKnownRandom,
      compactActionQuantile,
      seededCautiousExploration,
      opponentPredictTarget,
    )
  })
  return { opponentHands, styles, firstOpponentActions, seed: sampleSeed, compactOpponentTemperature: compactOpponentFuture ? compactOpponentTemperature : null, compactOpponentContextual, compactOpponentTarget, compactOpponentVersion2, deterministicKnownCautious, deterministicKnownRandom, deterministicTempo, seededCautiousExploration }
}

function rolloutOpponentChoice(observation, determinization, hand, player, rows, seenCards, scores, rng) {
  const strategy = observation.opponentStrategies?.[player - 1]
  if (determinization.deterministicKnownCautious && strategy === 'cautious') {
    return cautiousChoice(hand, rows, { ...observation, rows, hand, seenCards, scores }, null)
  }
  if (determinization.seededCautiousExploration != null && strategy === 'cautious') {
    const turnOffset = observation.hand.length - hand.length
    const seeded = seededCautiousChoice(observation, hand, player, determinization.seededCautiousExploration, turnOffset, rows, seenCards, scores)
    if (seeded != null) return seeded
  }
  if (determinization.deterministicKnownRandom && strategy === 'random') {
    const turnOffset = observation.hand.length - hand.length
    const exact = knownRandomChoice(observation, hand, player, turnOffset)
    if (exact != null) return exact
  }
  if (determinization.deterministicTempo && determinization.styles[player - 1] === 'tempo') {
    return tempoChoice(hand, rows, { ...observation, rows, hand, seenCards, scores }, null)
  }
  const targetMatches = determinization.compactOpponentTarget === 'both' || determinization.compactOpponentTarget === strategy
  const selectedTemperature = determinization.compactOpponentTemperature && typeof determinization.compactOpponentTemperature === 'object' ? determinization.compactOpponentTemperature[strategy] : determinization.compactOpponentTemperature
  if (selectedTemperature != null && targetMatches && (strategy === 'champion' || strategy === 'external_mcs')) {
    const proxyObservation = simulationPerspective({ ...observation, rows, seenCards, scores }, hand, player, determinization.seed)
    return chooseCompactOpponentCard(proxyObservation, opponentProxyModel(strategy, determinization.compactOpponentContextual, determinization.compactOpponentVersion2), rng, selectedTemperature)
  }
  return policyChoice(determinization.styles[player - 1], hand, rows, { ...observation, rows, hand, seenCards, scores }, rng)
}

function simulateCandidate(candidate, observation, determinization, legacy = false, scoring = {}) {
  const hands = [[...observation.hand], ...determinization.opponentHands.map((hand) => [...hand])]
  const scores = observation.scores ? [...observation.scores] : Array(observation.playerCount).fill(0)
  const startingScores = [...scores]
  let rows = cloneRows(observation.rows)
  const seenCards = [...observation.seenCards]
  const playedCards = (observation.playedCards || Array.from({ length: observation.playerCount }, () => []))
    .map((cards) => [...cards])
  // A determinization is deliberately shared by every root candidate.  Keep
  // the rollout policy's random stream shared as well when CRN is enabled;
  // otherwise candidate-specific seeds re-introduce most of the comparison
  // noise that common information sets were meant to remove.
  const rolloutSeed = scoring.commonRandomNumbers
    ? determinization.seed ^ 0x43524e
    : determinization.seed ^ (candidate * 0x9e3779b1)
  const rng = mulberry32(rolloutSeed)

  let actions = [candidate, ...determinization.firstOpponentActions]
  actions.forEach((card, player) => removeCard(hands[player], card))
  let result = playCards(rows, actions)
  rows = result.rows
  result.penalties.forEach((penalty, player) => { scores[player] += penalty })
  const immediatePenalty = result.penalties[0]
  seenCards.push(...actions)
  actions.forEach((card, player) => { playedCards[player].push(card) })
  let learnedLeafCost = null
  let learnedWinProbability = null
  if ((scoring.leafValueWeight || scoring.leafWinWeight) && hands[0].length > 0) {
    const leafObservation = { ...observation, rows, hand: hands[0], seenCards, scores, playedCards }
    if (scoring.leafValueWeight) {
      const leafValueModel = scoring.leafValueModel === 'kr10k' ? realArenaKrValuePolicyV2Model : realArenaValuePolicyV2Model
      const leafValue = evaluatePolicyV2Cards(leafObservation, leafValueModel)[0].value
      learnedLeafCost = immediatePenalty - leafValue
    }
    if (scoring.leafWinWeight) {
      const winValueModel = scoring.leafWinModel === 'new'
        ? realArenaWinValueNewPolicyV2Model
        : scoring.leafWinModel === 'mixed'
          ? realArenaWinValueMixedPolicyV2Model
          : realArenaWinValuePolicyV2Model
      const winLogit = evaluatePolicyV2Cards(leafObservation, winValueModel)[0].value
      learnedWinProbability = 1 / (1 + Math.exp(-winLogit))
    }
  }

  const blendLearnedLeaf = (outcome) => {
    let cost = outcome.cost
    if (learnedLeafCost != null) {
      const weight = Math.max(0, Math.min(1, scoring.leafValueWeight || 0))
      cost = cost * (1 - weight) + learnedLeafCost * weight
    }
    if (learnedWinProbability != null) cost -= learnedWinProbability * scoring.leafWinWeight
    return cost === outcome.cost ? outcome : { ...outcome, cost }
  }

  const terminalCost = (terminalScores) => {
    const focalPenalty = terminalScores[0] - startingScores[0]
    const opponentDeltas = terminalScores.slice(1).map((score, index) => score - startingScores[index + 1])
    const averageOpponent = opponentDeltas.reduce((sum, value) => sum + value, 0) / Math.max(1, opponentDeltas.length)
    const rankPressure = terminalScores.slice(1).reduce((sum, score) => sum + (terminalScores[0] > score ? 1 : terminalScores[0] === score ? 0.4 : 0), 0)
    const rankWeight = scoring.rankWeight ?? (legacy ? 0.18 : 0.42)
    const relativeWeight = scoring.relativeWeight ?? 0.06
    const minimumScore = Math.min(...terminalScores)
    const winnerCount = terminalScores.filter((score) => score === minimumScore).length
    const winShare = terminalScores[0] === minimumScore ? 1 / winnerCount : 0
    return {
      cost: focalPenalty + rankPressure * rankWeight + Math.max(0, focalPenalty - averageOpponent) * relativeWeight + (1 - winShare) * (scoring.winWeight ?? 0),
      focalPenalty,
    }
  }

  // In the small endgame the focal player's continuation tree is tiny
  // (at most 4! leaves).  Solving it per determinization is substantially
  // less biased than committing every root action to one myopic rollout.
  if (scoring.exactEndgameThreshold && hands[0].length <= scoring.exactEndgameThreshold) {
    const solve = (nodeHands, nodeRows, nodeScores, nodeSeen, depth) => {
      if (nodeHands[0].length === 0) return terminalCost(nodeScores)
      const opponentActions = nodeHands.slice(1).map((hand, opponentIndex) => {
        const player = opponentIndex + 1
        const nodeRng = mulberry32(determinization.seed ^ Math.imul(depth + 1, 0x45d9f3b) ^ Math.imul(player + 1, 0x119de1f3))
        return rolloutOpponentChoice(observation, determinization, hand, player, nodeRows, nodeSeen, nodeScores, nodeRng)
      })
      let best = null
      for (const focalCard of nodeHands[0]) {
        const nextHands = nodeHands.map((hand) => [...hand])
        const nextActions = [focalCard, ...opponentActions]
        nextActions.forEach((card, player) => removeCard(nextHands[player], card))
        const nextResult = playCards(nodeRows, nextActions)
        const nextScores = [...nodeScores]
        nextResult.penalties.forEach((penalty, player) => { nextScores[player] += penalty })
        const branch = solve(nextHands, nextResult.rows, nextScores, [...nodeSeen, ...nextActions], depth + 1)
        if (!best || branch.cost < best.cost || (branch.cost === best.cost && focalCard < best.card)) best = { ...branch, card: focalCard }
      }
      return best
    }
    const solved = solve(hands, rows, scores, seenCards, 0)
    return blendLearnedLeaf({ cost: solved.cost, focalPenalty: solved.focalPenalty, immediatePenalty })
  }

  if (scoring.beamEndgameWidth && hands[0].length <= (scoring.beamEndgameThreshold || 5)) {
    let frontier = [{ hands, rows, scores, seenCards }]
    let depth = 0
    while (frontier[0].hands[0].length > 0) {
      const expanded = []
      for (const node of frontier) {
        const opponentActions = node.hands.slice(1).map((hand, opponentIndex) => {
          const player = opponentIndex + 1
          const nodeRng = mulberry32(determinization.seed ^ Math.imul(depth + 1, 0x45d9f3b) ^ Math.imul(player + 1, 0x119de1f3))
          return rolloutOpponentChoice(observation, determinization, hand, player, node.rows, node.seenCards, node.scores, nodeRng)
        })
        for (const focalCard of node.hands[0]) {
          const nextHands = node.hands.map((hand) => [...hand])
          const nextActions = [focalCard, ...opponentActions]
          nextActions.forEach((card, player) => removeCard(nextHands[player], card))
          const nextResult = playCards(node.rows, nextActions)
          const nextScores = [...node.scores]
          nextResult.penalties.forEach((penalty, player) => { nextScores[player] += penalty })
          const nextSeen = [...node.seenCards, ...nextActions]
          const partial = terminalCost(nextScores)
          const continuation = nextHands[0].length
            ? rankedChoices(nextHands[0], nextResult.rows, { ...observation, rows: nextResult.rows, hand: nextHands[0], seenCards: nextSeen, scores: nextScores })[0].cost * (scoring.beamHeuristicWeight ?? 0.35)
            : 0
          expanded.push({ hands: nextHands, rows: nextResult.rows, scores: nextScores, seenCards: nextSeen, estimate: partial.cost + continuation, card: focalCard })
        }
      }
      expanded.sort((left, right) => left.estimate - right.estimate || left.card - right.card)
      frontier = expanded.slice(0, scoring.beamEndgameWidth)
      depth += 1
    }
    const solved = frontier.map((node) => terminalCost(node.scores)).sort((left, right) => left.cost - right.cost)[0]
    return blendLearnedLeaf({ cost: solved.cost, focalPenalty: solved.focalPenalty, immediatePenalty })
  }

  while (hands[0].length > 0) {
    actions = hands.map((hand, player) => {
      const rolloutObservation = { ...observation, rows, hand, seenCards, scores, playedCards }
      if (player === 0 && scoring.focalStyle === 'bestResponse') return null
      if (player === 0) {
        if (scoring.focalStyle === 'random') return policyChoice('random', hand, rows, rolloutObservation, rng)
        if (scoring.focalStyle === 'greedy') return policyChoice('greedy', hand, rows, rolloutObservation, rng)
        if (scoring.focalStyle === 'tempo') return policyChoice('tempo', hand, rows, rolloutObservation, rng)
        if (scoring.focalStyle === 'policyV2') {
          const rolloutDepth = observation.hand.length - hand.length
          if (!scoring.focalPolicyDepth || rolloutDepth <= scoring.focalPolicyDepth) {
            return choosePolicyV2Card(rolloutObservation, policyV2Model).card
          }
        }
        if (scoring.focalStyle === 'awrPolicy') {
          const rolloutDepth = observation.hand.length - hand.length
          if (!scoring.focalPolicyDepth || rolloutDepth <= scoring.focalPolicyDepth) {
            return choosePolicyV2Card(rolloutObservation, realArenaAwrPolicyV2Model).card
          }
        }
        if (scoring.focalStyle === 'compactPolicy') {
          const rolloutDepth = observation.hand.length - hand.length
          if (!scoring.focalPolicyDepth || rolloutDepth <= scoring.focalPolicyDepth) {
            const focalModel = scoring.focalCompactVariant === 't05' ? compactFocalPolicyT05Model : compactFocalPolicyModel
            return chooseCompactOpponentCard(rolloutObservation, focalModel, null, 0)
          }
        }
        if (scoring.focalStyle === 'distilledPolicy') {
          const rolloutDepth = observation.hand.length - hand.length
          if (!scoring.focalPolicyDepth || rolloutDepth <= scoring.focalPolicyDepth) {
            return chooseCompactPolicyV2Card(rolloutObservation, compactFullHistoryPolicyModel).card
          }
        }
        if (scoring.focalStyle === 'counterfactualPolicy') {
          const rolloutDepth = observation.hand.length - hand.length
          const counterfactualFraction = scoring.focalCounterfactualFraction ?? 1
          const useCounterfactual = counterfactualFraction >= 1
            || mulberry32(determinization.seed ^ 0x46514346)() < counterfactualFraction
          if (useCounterfactual && (!scoring.focalPolicyDepth || rolloutDepth <= scoring.focalPolicyDepth)) {
            if (scoring.focalCounterfactualVariant === 't05') {
              return chooseCompactOpponentCard(rolloutObservation, compactCounterfactualPolicyT05Model, null, 0)
            }
            return chooseCounterfactualCard(rolloutObservation)
          }
        }
        if (scoring.focalStyle === 'neural') return chooseNeuralCard(rolloutObservation, neuralModelFor(rolloutObservation)).card
        return legacy ? greedyChoice(hand, rows) : cautiousChoice(hand, rows, rolloutObservation, rng)
      }
      return rolloutOpponentChoice(observation, determinization, hand, player, rows, seenCards, scores, rng)
    })
    if (scoring.focalStyle === 'bestResponse') {
      const rolloutObservation = { ...observation, rows, hand: hands[0], seenCards, scores, playedCards }
      actions[0] = hands[0].map((card) => {
        const trialActions = [card, ...actions.slice(1)]
        const trial = playCards(rows, trialActions)
        const remaining = hands[0].filter((value) => value !== card)
        const futureCost = remaining.length
          ? rankedChoices(remaining, trial.rows, { ...rolloutObservation, rows: trial.rows, hand: remaining, seenCards: [...seenCards, ...trialActions] })[0].cost
          : 0
        return { card, cost: trial.penalties[0] * (scoring.focalPenaltyWeight ?? 5) + futureCost }
      }).sort((left, right) => left.cost - right.cost || left.card - right.card)[0].card
    }
    actions.forEach((card, player) => removeCard(hands[player], card))
    result = playCards(rows, actions)
    rows = result.rows
    result.penalties.forEach((penalty, player) => { scores[player] += penalty })
    seenCards.push(...actions)
    actions.forEach((card, player) => { playedCards[player].push(card) })
  }

  const terminal = terminalCost(scores)
  return blendLearnedLeaf({
    cost: terminal.cost,
    focalPenalty: terminal.focalPenalty,
    immediatePenalty,
  })
}

function summarizeEvaluations(stats, sampleCount, champion, scoring = {}) {
  return stats.map((entry) => {
    const count = entry.costs.length || sampleCount
    const mean = entry.costs.reduce((sum, value) => sum + value, 0) / count
    const focalMean = entry.focalSum / count
    const variance = entry.costs.reduce((sum, value) => sum + (value - mean) ** 2, 0) / count
    const risk = Math.sqrt(Math.max(0, variance))
    const sortedWorstFirst = [...entry.costs].sort((left, right) => right - left)
    const tailCount = Math.max(1, Math.ceil(count * 0.2))
    const cvar = sortedWorstFirst.slice(0, tailCount).reduce((sum, value) => sum + value, 0) / tailCount
    const disasterRate = entry.disasters / count
    const riskScale = scoring.riskScale ?? 1
    const rescueUncertainty = entry.rescue ? (scoring.rescueConfidenceZ ?? 0) * risk / Math.sqrt(count) : 0
    const immediateMean = entry.immediateSum / count
    const utility = champion ? mean + immediateMean * (scoring.immediateWeight ?? 0) + risk * 0.055 * riskScale + (cvar - mean) * 0.11 * riskScale + disasterRate * 0.35 * riskScale + rescueUncertainty : mean + risk * 0.1 + disasterRate * 0.55
    return {
      card: entry.card,
      utility,
      expectedPenalty: focalMean,
      immediatePenalty: immediateMean,
      risk,
      cvar,
      disasterRate,
      samples: count,
      rescue: Boolean(entry.rescue),
    }
  }).sort((left, right) => left.utility - right.utility || left.card - right.card)
}

function searchChoice(observation, champion, options = {}) {
  const baseSamples = Math.max(8, observation.samples || (champion ? 120 : 60))
  const candidateCards = options.candidateCards || observation.hand
  const endgameMultiplier = observation.hand.length <= 3 && champion ? 1.45 : 1
  const budgetMultiplier = options.equalizeBudget ? observation.hand.length / candidateCards.length : 1
  const sampleCount = Math.max(1, Math.round(baseSamples * endgameMultiplier * budgetMultiplier))
  const seed = observation.seed || 0x6e696d6d
  const stats = candidateCards.map((card) => ({ card, costs: [], focalSum: 0, immediateSum: 0, disasters: 0 }))

  if (champion && options.rescueFraction && stats.length === 5) {
    const totalBudget = Math.max(stats.length, Math.round(baseSamples * endgameMultiplier * observation.hand.length))
    const maximumRescue = Math.max(1, totalBudget - (stats.length - 1))
    const rescueBudget = Math.min(maximumRescue, Math.max(4, Math.round(totalBudget * options.rescueFraction)))
    const primaryTotal = totalBudget - rescueBudget
    const primaryBase = Math.floor(primaryTotal / (stats.length - 1))
    const primaryRemainder = primaryTotal % (stats.length - 1)
    const allocations = stats.map((entry, index) => {
      if (index === stats.length - 1) {
        entry.rescue = true
        return rescueBudget
      }
      return primaryBase + (index < primaryRemainder ? 1 : 0)
    })
    const maximumSamples = Math.max(...allocations)
    for (let sample = 0; sample < maximumSamples; sample += 1) {
      const determinization = createDeterminization(observation, seed + sample * 7919, false, options.opponentProfile, options.historyAttempts, options.historyMode, options.opponentPredictSamples, options.compactOpponentTemperature, options.compactOpponentFuture, options.compactOpponentContextual, options.compactHistoryAttempts, options.compactOpponentTarget, options.compactOpponentVersion2, options.stratified ? sample : -1, seed, options.deterministicKnownCautious, options.deterministicKnownRandom, options.deterministicKnownCautiousRoot, options.compactStratifiedActions, options.deterministicTempo, options.seededCautiousExploration, options.balancedStratified, options.rotateStratifiedPhases, useHandPosteriorSample(options, sample), options.opponentPredictTarget)
      stats.forEach((entry, index) => {
        if (sample >= allocations[index]) return
        const outcome = simulateCandidate(entry.card, observation, determinization, false, options.scoring)
        entry.costs.push(outcome.cost)
        entry.focalSum += outcome.focalPenalty
        entry.immediateSum += outcome.immediatePenalty
        if (outcome.focalPenalty >= 8) entry.disasters += 1
      })
    }
  } else if (champion && options.successiveHalving && stats.length > (options.successiveFinalists ?? 2)) {
    const totalBudget = Math.max(stats.length, Math.round(baseSamples * endgameMultiplier * observation.hand.length))
    const finalistCount = Math.max(1, Math.min(stats.length - 1, options.successiveFinalists ?? 2))
    const stageOneSamples = Math.max(4, Math.floor(totalBudget / (stats.length + finalistCount)))
    for (let sample = 0; sample < stageOneSamples; sample += 1) {
      const determinization = createDeterminization(observation, seed + sample * 7919, false, options.opponentProfile, options.historyAttempts, options.historyMode, options.opponentPredictSamples, options.compactOpponentTemperature, options.compactOpponentFuture, options.compactOpponentContextual, options.compactHistoryAttempts, options.compactOpponentTarget, options.compactOpponentVersion2, options.stratified ? sample : -1, seed, options.deterministicKnownCautious, options.deterministicKnownRandom, options.deterministicKnownCautiousRoot, options.compactStratifiedActions, options.deterministicTempo, options.seededCautiousExploration, options.balancedStratified, options.rotateStratifiedPhases, useHandPosteriorSample(options, sample), options.opponentPredictTarget)
      stats.forEach((entry) => {
        const outcome = simulateCandidate(entry.card, observation, determinization, false, options.scoring)
        entry.costs.push(outcome.cost)
        entry.focalSum += outcome.focalPenalty
        entry.immediateSum += outcome.immediatePenalty
        if (outcome.focalPenalty >= 8) entry.disasters += 1
      })
    }
    const finalistCards = new Set(summarizeEvaluations(stats, stageOneSamples, true, options.scoring).slice(0, finalistCount).map((entry) => entry.card))
    const finalists = stats.filter((entry) => finalistCards.has(entry.card))
    const remainingBudget = Math.max(0, totalBudget - stageOneSamples * stats.length)
    const extraSamples = Math.floor(remainingBudget / finalists.length)
    for (let extra = 0; extra < extraSamples; extra += 1) {
      const sample = stageOneSamples + extra
      const determinization = createDeterminization(observation, seed + sample * 7919, false, options.opponentProfile, options.historyAttempts, options.historyMode, options.opponentPredictSamples, options.compactOpponentTemperature, options.compactOpponentFuture, options.compactOpponentContextual, options.compactHistoryAttempts, options.compactOpponentTarget, options.compactOpponentVersion2, options.stratified ? sample : -1, seed, options.deterministicKnownCautious, options.deterministicKnownRandom, options.deterministicKnownCautiousRoot, options.compactStratifiedActions, options.deterministicTempo, options.seededCautiousExploration, options.balancedStratified, options.rotateStratifiedPhases, useHandPosteriorSample(options, sample), options.opponentPredictTarget)
      finalists.forEach((entry) => {
        const outcome = simulateCandidate(entry.card, observation, determinization, false, options.scoring)
        entry.costs.push(outcome.cost)
        entry.focalSum += outcome.focalPenalty
        entry.immediateSum += outcome.immediatePenalty
        if (outcome.focalPenalty >= 8) entry.disasters += 1
      })
    }
    stats.splice(0, stats.length, ...finalists)
  } else if (champion) {
    // Common random information sets reduce comparison noise between root actions.
    for (let sample = 0; sample < sampleCount; sample += 1) {
      const determinization = createDeterminization(observation, seed + sample * 7919, false, options.opponentProfile, options.historyAttempts, options.historyMode, options.opponentPredictSamples, options.compactOpponentTemperature, options.compactOpponentFuture, options.compactOpponentContextual, options.compactHistoryAttempts, options.compactOpponentTarget, options.compactOpponentVersion2, options.stratified ? sample : -1, seed, options.deterministicKnownCautious, options.deterministicKnownRandom, options.deterministicKnownCautiousRoot, options.compactStratifiedActions, options.deterministicTempo, options.seededCautiousExploration, options.balancedStratified, options.rotateStratifiedPhases, useHandPosteriorSample(options, sample), options.opponentPredictTarget)
      stats.forEach((entry) => {
        const outcome = simulateCandidate(entry.card, observation, determinization, false, options.scoring)
        entry.costs.push(outcome.cost)
        entry.focalSum += outcome.focalPenalty
        entry.immediateSum += outcome.immediatePenalty
        if (outcome.focalPenalty >= 8) entry.disasters += 1
      })
    }
  } else {
    stats.forEach((entry, cardIndex) => {
      for (let sample = 0; sample < sampleCount; sample += 1) {
        const determinization = createDeterminization(observation, seed + cardIndex * 1000003 + sample * 7919, true)
        const outcome = simulateCandidate(entry.card, observation, determinization, true)
        entry.costs.push(outcome.cost)
        entry.focalSum += outcome.focalPenalty
        entry.immediateSum += outcome.immediatePenalty
        if (outcome.focalPenalty >= 8) entry.disasters += 1
      }
    })
  }
  let evaluations = summarizeEvaluations(stats, sampleCount, champion, options.scoring)
  if (champion && options.scoring?.sampleRankWeight > 0 && stats.length > 1) {
    const commonSamples = Math.min(...stats.map((entry) => entry.costs.length))
    const rankSums = new Map(stats.map((entry) => [entry.card, 0]))
    for (let sample = 0; sample < commonSamples; sample += 1) {
      const ordered = stats
        .map((entry) => ({ card: entry.card, cost: entry.costs[sample] }))
        .sort((left, right) => left.cost - right.cost || left.card - right.card)
      for (let start = 0; start < ordered.length;) {
        let end = start + 1
        while (end < ordered.length && ordered[end].cost === ordered[start].cost) end += 1
        const averageRank = (start + end - 1) / 2
        for (let index = start; index < end; index += 1) {
          rankSums.set(ordered[index].card, rankSums.get(ordered[index].card) + averageRank)
        }
        start = end
      }
    }
    const pairedRank = new Map([...rankSums].map(([card, sum]) => [card, sum / Math.max(1, commonSamples)]))
    const meanOrder = new Map(evaluations.map((entry, index) => [entry.card, index]))
    const pairedOrder = new Map([...evaluations]
      .sort((left, right) => pairedRank.get(left.card) - pairedRank.get(right.card) || left.card - right.card)
      .map((entry, index) => [entry.card, index]))
    const weight = Math.max(0, Math.min(1, options.scoring.sampleRankWeight))
    evaluations = evaluations.map((entry) => ({
      ...entry,
      meanUtility: entry.utility,
      pairedRank: pairedRank.get(entry.card),
      utility: meanOrder.get(entry.card) * (1 - weight) + pairedOrder.get(entry.card) * weight,
    })).sort((left, right) => left.utility - right.utility || left.card - right.card)
  }
  if (options.neuralPrior && options.rankPriorWeight > 0) {
    const searchRank = new Map(evaluations.map((entry, index) => [entry.card, index]))
    const neuralOrder = [...evaluations].sort((left, right) => (
      options.neuralPrior.get(left.card) - options.neuralPrior.get(right.card) || left.card - right.card
    ))
    const neuralRank = new Map(neuralOrder.map((entry, index) => [entry.card, index]))
    const rankWeight = options.rankPriorWeight
    evaluations = evaluations.map((entry) => ({
      ...entry,
      searchUtility: entry.utility,
      neuralUtility: options.neuralPrior.get(entry.card),
      utility: searchRank.get(entry.card) * (1 - rankWeight) + neuralRank.get(entry.card) * rankWeight,
    })).sort((left, right) => left.utility - right.utility || left.card - right.card)
  } else if (options.neuralPrior) {
    const priorWeight = options.priorWeight ?? 0.18
    evaluations = evaluations.map((entry) => {
      const neuralUtility = options.neuralPrior.get(entry.card)
      return {
        ...entry,
        searchUtility: entry.utility,
        neuralUtility,
        utility: entry.utility * (1 - priorWeight) + neuralUtility * priorWeight,
      }
    }).sort((left, right) => left.utility - right.utility || left.card - right.card)
  }
  return { card: evaluations[0].card, evaluations, sampleCount }
}

function neuralHybridChoice(observation, priorWeight = 0.10, classicNeuralThreshold = 11, candidateFraction = null, useClassicBook = true, scoring = {}, opponentProfile = 'mixed', confidenceThreshold = 0, model = null, ensembleModel = null, openingNeuralThreshold = 11, successiveHalving = false, minimumCandidates = 3, includeHeuristic = true, historyAttempts = 0, historyMode = 'exact', opponentPredictSamples = 0) {
  const classicMode = observation.deckSize > observation.playerCount * 10 + 4
  const selectedModel = model || neuralModelFor(observation)
  if (classicMode && observation.hand.length >= classicNeuralThreshold) return chooseNeuralCard(observation, selectedModel)
  if (!classicMode && observation.hand.length >= openingNeuralThreshold) return chooseNeuralCard(observation, selectedModel)
  if (classicMode && useClassicBook && observation.hand.length >= 8) {
    const evaluations = rankedChoices(observation.hand, observation.rows, observation).map((entry) => ({ ...entry, utility: entry.cost, risk: 0, disasterRate: 0 }))
    return { card: evaluations[0].card, evaluations, openingBook: true, neuralGate: true }
  }
  let neuralEvaluations = evaluateNeuralCards(observation, selectedModel)
  if (ensembleModel) {
    const secondary = evaluateNeuralCards(observation, ensembleModel)
    const primaryRank = new Map(neuralEvaluations.map((entry, index) => [entry.card, index]))
    const secondaryRank = new Map(secondary.map((entry, index) => [entry.card, index]))
    neuralEvaluations = neuralEvaluations.map((entry) => ({
      ...entry,
      utility: (primaryRank.get(entry.card) + secondaryRank.get(entry.card)) / 2,
    })).sort((left, right) => left.utility - right.utility || left.card - right.card)
  }
  const neuralMargin = neuralEvaluations.length > 1 ? neuralEvaluations[1].utility - neuralEvaluations[0].utility : Infinity
  const effectivePriorWeight = neuralMargin >= confidenceThreshold ? priorWeight : 0
  if (observation.hand.length <= 3) {
    return searchChoice(observation, true, {
      neuralPrior: new Map(neuralEvaluations.map((entry) => [entry.card, entry.utility])),
      priorWeight: effectivePriorWeight,
      successiveHalving,
      historyAttempts,
      historyMode,
      opponentPredictSamples,
      scoring,
      opponentProfile,
    })
  }
  const fraction = candidateFraction ?? (classicMode || observation.playerCount === 3 ? 1 : 0.5)
  const keep = Math.min(observation.hand.length, Math.max(minimumCandidates, Math.ceil(observation.hand.length * fraction)))
  const candidateCards = neuralEvaluations.slice(0, keep).map((entry) => entry.card)
  if (includeHeuristic) {
    const heuristicCard = rankedChoices(observation.hand, observation.rows, observation)[0].card
    if (!candidateCards.includes(heuristicCard)) candidateCards[candidateCards.length - 1] = heuristicCard
  }
  return searchChoice(observation, true, {
    candidateCards,
    equalizeBudget: true,
    neuralPrior: new Map(neuralEvaluations.map((entry) => [entry.card, entry.utility])),
    priorWeight: effectivePriorWeight,
    scoring,
    opponentProfile,
    successiveHalving,
    historyAttempts,
    historyMode,
    opponentPredictSamples,
  })
}

function policyV2HybridChoice(observation, strategy) {
  if (observation.deckSize !== 54 || observation.playerCount !== 5) return neuralHybridChoice(observation)
  // Parse tunable suffixes only after the strategy id.  Otherwise the
  // literal version marker in `neural_hybrid_v2` is mistaken for `_v2`.
  const strategyOptions = strategy.slice('neural_hybrid_v2'.length)
  let evaluations = evaluatePolicyV2Cards(observation, policyV2Model)
  const primaryEvaluations = evaluations
  const weightMatch = strategy.match(/_p(\d+)/)
  const priorStartMatch = strategy.match(/_ps([0-9])(?:_|$)/)
  const priorEndMatch = strategy.match(/_py([1-9])(?:_|$)/)
  const rankPriorMatch = strategy.match(/_nr(\d+)(?:_|$)/)
  const candidateMatch = strategy.match(/_k(\d+)/)
  const gateMatch = strategy.match(/_g(\d+)/)
  const teacherStartMatch = strategy.match(/_t3s([0-9])(?:_|$)/)
  const teacherEndMatch = strategy.match(/_t3y([1-9])(?:_|$)/)
  const policyTurn = 10 - observation.hand.length
  const configuredPriorWeight = weightMatch ? Number(weightMatch[1]) / 100 : 0.10
  const priorWeight = (priorStartMatch && policyTurn < Number(priorStartMatch[1])) || (priorEndMatch && policyTurn >= Number(priorEndMatch[1]))
    ? 0
    : configuredPriorWeight
  const candidateFraction = candidateMatch ? Number(candidateMatch[1]) / 100 : 0.50
  const gate = gateMatch ? Number(gateMatch[1]) : 11
  const ensembleMatch = strategy.match(/_e(\d+)/)
  const ensembleStartMatch = strategy.match(/_es([0-9])(?:_|$)/)
  const ensembleEndMatch = strategy.match(/_ee([1-9])(?:_|$)/)
  let legacyEvaluations = null
  const currentTurn = policyTurn
  const teacherCandidateActive = strategy.includes('_t3')
    && (!teacherStartMatch || currentTurn >= Number(teacherStartMatch[1]))
    && (!teacherEndMatch || currentTurn < Number(teacherEndMatch[1]))
  const ensembleActive = (!ensembleStartMatch || currentTurn >= Number(ensembleStartMatch[1])) && (!ensembleEndMatch || currentTurn < Number(ensembleEndMatch[1]))
  if (ensembleMatch && ensembleActive) {
    const legacyWeight = Math.max(0, Math.min(1, Number(ensembleMatch[1]) / 100))
    legacyEvaluations = evaluateNeuralCards(observation, neuralModelFor(observation))
    const v2Rank = new Map(evaluations.map((entry, index) => [entry.card, index]))
    const legacyRank = new Map(legacyEvaluations.map((entry, index) => [entry.card, index]))
    evaluations = evaluations.map((entry) => ({
      ...entry,
      policyV2Utility: entry.utility,
      utility: v2Rank.get(entry.card) * (1 - legacyWeight) + legacyRank.get(entry.card) * legacyWeight,
    })).sort((left, right) => left.utility - right.utility || left.card - right.card)
  }
  const counterfactualRootMatch = strategy.match(/_qr(\d{1,3})(?:_|$)/)
  if (counterfactualRootMatch) {
    const counterfactualWeight = Math.max(0, Math.min(1, Number(counterfactualRootMatch[1]) / 100))
    const counterfactualEvaluations = evaluateCounterfactualCards(observation)
    const policyRank = new Map(evaluations.map((entry, index) => [entry.card, index]))
    const counterfactualRank = new Map(counterfactualEvaluations.map((entry, index) => [entry.card, index]))
    evaluations = evaluations.map((entry) => ({
      ...entry,
      policyV2Utility: entry.utility,
      counterfactualUtility: counterfactualEvaluations.find((candidate) => candidate.card === entry.card)?.utility,
      utility: policyRank.get(entry.card) * (1 - counterfactualWeight) + counterfactualRank.get(entry.card) * counterfactualWeight,
    })).sort((left, right) => left.utility - right.utility || left.card - right.card)
  }
  if (observation.hand.length >= gate) return { card: evaluations[0].card, evaluations, policyV2: true }
  const scheduleMatch = strategy.match(/_q([1-9]{7})(?:_|$)/)
  const scheduledKeep = scheduleMatch && observation.hand.length >= 4
    ? Number(scheduleMatch[1][10 - observation.hand.length])
    : null
  const candidateExpandMatch = strategy.match(/_kx(\d+)(?:_|$)/)
  const candidateShrinkMatch = strategy.match(/_kv(\d+)(?:_|$)/)
  const policyMargin = evaluations.length > 1 ? evaluations[1].utility - evaluations[0].utility : Infinity
  let configuredKeep = scheduledKeep ?? Math.max(3, Math.ceil(observation.hand.length * candidateFraction))
  if (candidateExpandMatch && policyMargin <= Number(candidateExpandMatch[1]) / 10) configuredKeep += 1
  if (candidateShrinkMatch && policyMargin >= Number(candidateShrinkMatch[1]) / 10) configuredKeep -= 1
  const keep = Math.min(observation.hand.length, Math.max(Math.min(3, observation.hand.length), configuredKeep))
  let candidateCards
  if (strategy.includes('_qu')) {
    const counterfactualEvaluations = evaluateCounterfactualCards(observation)
    const primaryCount = Math.max(1, Math.ceil(keep / 2))
    const counterfactualCount = Math.max(1, keep - primaryCount + 1)
    candidateCards = [...new Set([
      ...evaluations.slice(0, primaryCount).map((entry) => entry.card),
      ...counterfactualEvaluations.slice(0, counterfactualCount).map((entry) => entry.card),
    ])].slice(0, keep)
    for (const source of [evaluations, counterfactualEvaluations]) {
      for (const entry of source) {
        if (candidateCards.length >= keep) break
        if (!candidateCards.includes(entry.card)) candidateCards.push(entry.card)
      }
      if (candidateCards.length >= keep) break
    }
  } else if (strategy.includes('_ua')) {
    const awrEvaluations = evaluatePolicyV2Cards(observation, realArenaAwrPolicyV2Model)
    const each = Math.max(1, Math.ceil(keep / 2))
    candidateCards = [...new Set([
      ...evaluations.slice(0, each).map((entry) => entry.card),
      ...awrEvaluations.slice(0, each).map((entry) => entry.card),
    ])].slice(0, keep)
    for (const source of [evaluations, awrEvaluations]) {
      for (const entry of source) {
        if (candidateCards.length >= keep) break
        if (!candidateCards.includes(entry.card)) candidateCards.push(entry.card)
      }
      if (candidateCards.length >= keep) break
    }
  } else if (teacherCandidateActive && legacyEvaluations) {
    const teacherEvaluations = evaluatePolicyV2Cards(observation, distilledTeacherPolicyV2Model)
    candidateCards = [...new Set([
      ...evaluations.slice(0, 2).map((entry) => entry.card),
      legacyEvaluations[0].card,
      teacherEvaluations[0].card,
    ])].slice(0, keep)
    for (const source of [primaryEvaluations, evaluations, teacherEvaluations, legacyEvaluations]) {
      for (const entry of source) {
        if (candidateCards.length >= keep) break
        if (!candidateCards.includes(entry.card)) candidateCards.push(entry.card)
      }
      if (candidateCards.length >= keep) break
    }
  } else if (strategy.includes('_u') && legacyEvaluations) {
    const each = Math.max(1, Math.ceil(keep / 2))
    candidateCards = [...new Set([
      ...evaluations.slice(0, each).map((entry) => entry.card),
      ...legacyEvaluations.slice(0, each).map((entry) => entry.card),
    ])].slice(0, keep)
    for (const entry of evaluations) {
      if (candidateCards.length >= keep) break
      if (!candidateCards.includes(entry.card)) candidateCards.push(entry.card)
    }
  } else {
    candidateCards = evaluations.slice(0, keep).map((entry) => entry.card)
  }
  const heuristicCard = rankedChoices(observation.hand, observation.rows, observation)[0].card
  const heuristicAddEndMatch = strategy.match(/_hay([1-9])(?:_|$)/)
  const heuristicAddActive = strategy.includes('_ha')
    && (!heuristicAddEndMatch || currentTurn < Number(heuristicAddEndMatch[1]))
  if (!strategy.includes('_h0') && !candidateCards.includes(heuristicCard)) {
    if (heuristicAddActive) candidateCards.push(heuristicCard)
    else candidateCards[candidateCards.length - 1] = heuristicCard
  }
  const counterfactualAppendEndMatch = strategy.match(/_qay([1-9])(?:_|$)/)
  const counterfactualAppendActive = strategy.includes('_qa')
    && (!counterfactualAppendEndMatch || currentTurn < Number(counterfactualAppendEndMatch[1]))
  if (counterfactualAppendActive) {
    const counterfactualCard = evaluateCounterfactualCards(observation)[0].card
    if (!candidateCards.includes(counterfactualCard)) candidateCards.push(counterfactualCard)
  }
  const knownArena = observation.opponentStrategies?.includes('champion') && observation.opponentStrategies?.includes('external_mcs')
  const exactProfileMatch = strategy.match(/_zx([cgtr])([cgtr])(?:_|$)/)
  const profileStartMatch = strategy.match(/_zs([0-9])(?:_|$)/)
  const profileEndMatch = strategy.match(/_zy([1-9])(?:_|$)/)
  const exactProfileActive = exactProfileMatch
    && (!profileStartMatch || currentTurn >= Number(profileStartMatch[1]))
    && (!profileEndMatch || currentTurn < Number(profileEndMatch[1]))
  const focalStartMatch = strategy.match(/_fs([0-9])(?:_|$)/)
  const focalEndMatch = strategy.match(/_fy([1-9])(?:_|$)/)
  const focalMixMatch = strategy.match(/_fm([1-8])([1-3])([1-3])(?:_|$)/)
  const focalStyleEnabled = (!focalStartMatch || currentTurn >= Number(focalStartMatch[1])) && (!focalEndMatch || currentTurn < Number(focalEndMatch[1]))
  const mixedFocalStyle = focalMixMatch
    ? currentTurn < Number(focalMixMatch[1]) ? 'compactPolicy' : 'distilledPolicy'
    : undefined
  const focalStyle = !focalStyleEnabled ? undefined : mixedFocalStyle || (strategy.includes('_fq') ? 'counterfactualPolicy' : strategy.includes('_fa') ? 'awrPolicy' : strategy.includes('_fd') ? 'distilledPolicy' : strategy.includes('_fc') ? 'compactPolicy' : strategy.includes('_fp') ? 'policyV2' : strategy.includes('_fb') ? 'bestResponse' : strategy.includes('_fg') ? 'greedy' : strategy.includes('_ft') ? 'tempo' : strategy.includes('_fr') ? 'random' : undefined)
  const rankMatch = strategy.match(/_r(\d+)/)
  const relativeMatch = strategy.match(/_d(\d+)/)
  const riskMatch = strategyOptions.match(/_v(\d+)(?:_|$)/)
  const focalPenaltyMatch = strategy.match(/_fw(\d+)/)
  const focalPolicyDepthMatch = strategy.match(/_fp([1-3])(?:_|$)/)
  const focalAwrDepthMatch = strategy.match(/_fa([1-3])(?:_|$)/)
  const focalCounterfactualDepthMatch = strategy.match(/_fq([1-3])(?:_|$)/)
  const focalCounterfactualFractionMatch = strategy.match(/_qf(\d+)(?:_|$)/)
  const focalCompactDepthMatch = strategy.match(/_fc([1-3])(?:_|$)/)
  const focalDistilledDepthMatch = strategy.match(/_fd([1-3])(?:_|$)/)
  const focalMixedDepth = focalMixMatch
    ? Number(currentTurn < Number(focalMixMatch[1]) ? focalMixMatch[2] : focalMixMatch[3])
    : 0
  const immediateWeightMatch = strategy.match(/_iw(\d+)(?:_|$)/)
  const immediateStartMatch = strategy.match(/_is([0-9])(?:_|$)/)
  const immediateEndMatch = strategy.match(/_iy([1-9])(?:_|$)/)
  const immediateWeightActive = (!immediateStartMatch || currentTurn >= Number(immediateStartMatch[1]))
    && (!immediateEndMatch || currentTurn < Number(immediateEndMatch[1]))
  const leafValueMatch = strategy.match(/_lv(\d+)(?:_|$)/)
  const leafWinMatch = strategy.match(/_lw(\d+)(?:_|$)/)
  const exactEndgameMatch = strategy.match(/_ex([1-5])(?:_|$)/)
  const compactOpponentMatch = strategy.match(/_cp(\d+)(?:_|$)/)
  const compactChampionMatch = strategy.match(/_cc(\d+)(?:_|$)/)
  const compactExternalMatch = strategy.match(/_ce(\d+)(?:_|$)/)
  const compactStartMatch = strategy.match(/_cs([0-9])(?:_|$)/)
  const compactTemperatureScaleMatch = strategy.match(/_ts(\d+)(?:_|$)/)
  const winWeightMatch = strategy.match(/_w(\d+)(?:_|$)/)
  const winStartMatch = strategy.match(/_ws([0-9])(?:_|$)/)
  const winEndMatch = strategy.match(/_wy([1-9])(?:_|$)/)
  const winWeightActive = (!winStartMatch || currentTurn >= Number(winStartMatch[1]))
    && (!winEndMatch || currentTurn < Number(winEndMatch[1]))
  const rankStartMatch = strategy.match(/_rs([0-9])(?:_|$)/)
  const rankEndMatch = strategy.match(/_ry([1-9])(?:_|$)/)
  const rankWeightActive = (!rankStartMatch || currentTurn >= Number(rankStartMatch[1]))
    && (!rankEndMatch || currentTurn < Number(rankEndMatch[1]))
  const sampleRankMatch = strategy.match(/_sr(\d+)(?:_|$)/)
  const compactHistoryMatch = strategy.match(/_ph(\d+)(?:_|$)/)
  const historyMatch = strategy.match(/_bh(\d+)?(?:_|$)/)
  const historyStartMatch = strategy.match(/_bt(\d+)/)
  const opponentPredictMatch = strategy.match(/_xp(\d+)(?:_|$)/)
  const opponentPredictStartMatch = strategy.match(/_xs([0-9])(?:_|$)/)
  const opponentPredictEndMatch = strategy.match(/_xy([1-9])(?:_|$)/)
  const handPosteriorMatch = strategy.match(/_hp([0-9])?(?:_|$)/)
  const handPosteriorFractionMatch = strategy.match(/_hm(\d+)(?:_|$)/)
  const beamEndgameMatch = strategy.match(/_bm(\d+)(?:_|$)/)
  const beamEndgameThresholdMatch = strategy.match(/_bn([5-6])(?:_|$)/)
  const beamHeuristicMatch = strategy.match(/_bl(\d+)(?:_|$)/)
  const rescueBudgetMatch = strategy.match(/_rb(\d+)(?:_|$)/)
  const rescueConfidenceMatch = strategy.match(/_rz(\d+)(?:_|$)/)
  const seededCautiousMatch = strategy.match(/_kq(\d+)?(?:_|$)/)
  const successiveFinalistsMatch = strategy.match(/_sf([2-5])(?:_|$)/)
  let compactOpponentTemperature = compactChampionMatch || compactExternalMatch
    ? {
        champion: compactChampionMatch ? Number(compactChampionMatch[1]) / 10 : null,
        external_mcs: compactExternalMatch ? Number(compactExternalMatch[1]) / 10 : null,
      }
    : compactOpponentMatch ? Number(compactOpponentMatch[1]) / 10 : null
  if (strategy.includes('_td')) {
    const turn = 10 - observation.hand.length
    const scale = compactTemperatureScaleMatch ? Number(compactTemperatureScaleMatch[1]) / 10 : 1
    const championSchedule = [2, 1, 1, 1, 1, 1, 1, 1, 1, 0.2]
    const externalSchedule = [1.5, 1, 1, 1, 1, 1, 0.7, 0.7, 1.5, 0.2]
    compactOpponentTemperature = { champion: championSchedule[turn] * scale, external_mcs: externalSchedule[turn] * scale }
  }
  if (compactStartMatch && 10 - observation.hand.length < Number(compactStartMatch[1])) compactOpponentTemperature = null
  const historyStart = historyStartMatch ? Number(historyStartMatch[1]) : 1
  const historyAttempts = historyMatch && (observation.history?.length || 0) >= historyStart ? Number(historyMatch[1] || 4) : 0
  const opponentPredictActive = (!opponentPredictStartMatch || currentTurn >= Number(opponentPredictStartMatch[1]))
    && (!opponentPredictEndMatch || currentTurn < Number(opponentPredictEndMatch[1]))
  const opponentPredictSamples = opponentPredictMatch && opponentPredictActive ? Math.max(1, Number(opponentPredictMatch[1])) : 0
  let searchResult = searchChoice(observation, true, {
    candidateCards,
    equalizeBudget: true,
    neuralPrior: new Map(evaluations.map((entry) => [entry.card, entry.utility])),
    priorWeight,
    rankPriorWeight: rankPriorMatch ? Number(rankPriorMatch[1]) / 100 : 0,
    scoring: {
      focalStyle,
      commonRandomNumbers: strategy.includes('_crn'),
      exactEndgameThreshold: exactEndgameMatch ? Number(exactEndgameMatch[1]) : 0,
      winWeight: winWeightMatch && winWeightActive ? Number(winWeightMatch[1]) : 0,
      sampleRankWeight: sampleRankMatch ? Number(sampleRankMatch[1]) / 100 : 0,
      beamEndgameWidth: beamEndgameMatch ? Number(beamEndgameMatch[1]) : 0,
      beamEndgameThreshold: beamEndgameThresholdMatch ? Number(beamEndgameThresholdMatch[1]) : 5,
      beamHeuristicWeight: beamHeuristicMatch ? Number(beamHeuristicMatch[1]) / 100 : undefined,
      rescueConfidenceZ: rescueConfidenceMatch ? Number(rescueConfidenceMatch[1]) / 10 : undefined,
      rankWeight: rankMatch && rankWeightActive ? Number(rankMatch[1]) / 100 : undefined,
      relativeWeight: relativeMatch ? Number(relativeMatch[1]) / 100 : undefined,
      // Preserve the released v2 behavior (the old parser accidentally used
      // the version number as 0.02) while allowing explicit risk suffixes.
      riskScale: riskMatch ? Number(riskMatch[1]) / 100 : 0.02,
      focalPenaltyWeight: focalPenaltyMatch ? Number(focalPenaltyMatch[1]) : undefined,
      focalPolicyDepth: focalMixedDepth || (focalPolicyDepthMatch ? Number(focalPolicyDepthMatch[1]) : focalCounterfactualDepthMatch ? Number(focalCounterfactualDepthMatch[1]) : focalAwrDepthMatch ? Number(focalAwrDepthMatch[1]) : focalCompactDepthMatch ? Number(focalCompactDepthMatch[1]) : focalDistilledDepthMatch ? Number(focalDistilledDepthMatch[1]) : 0),
      focalCompactVariant: strategy.includes('_fct5') ? 't05' : 'default',
      focalCounterfactualVariant: strategy.includes('_fqt5') ? 't05' : 'default',
      focalCounterfactualFraction: focalCounterfactualFractionMatch ? Number(focalCounterfactualFractionMatch[1]) / 100 : 1,
      immediateWeight: immediateWeightMatch && immediateWeightActive ? Number(immediateWeightMatch[1]) / 100 : 0,
      leafValueWeight: leafValueMatch ? Number(leafValueMatch[1]) / 100 : 0,
      leafValueModel: strategy.includes('_lvk') ? 'kr10k' : 'legacy',
      leafWinWeight: leafWinMatch ? Number(leafWinMatch[1]) : 0,
      leafWinModel: strategy.includes('_lwn') ? 'new' : strategy.includes('_lwm') ? 'mixed' : 'old',
    },
    successiveHalving: strategy.includes('_sh') || Boolean(successiveFinalistsMatch),
    successiveFinalists: successiveFinalistsMatch ? Number(successiveFinalistsMatch[1]) : undefined,
    rescueFraction: rescueBudgetMatch ? Number(rescueBudgetMatch[1]) / 100 : 0,
    historyAttempts,
    historyMode: strategy.includes('_hs') ? 'seededCautious' : strategy.includes('_hc') ? 'cautious' : strategy.includes('_hr') ? 'rank' : 'exact',
    opponentPredictSamples,
    opponentPredictTarget: strategy.includes('_xpc') ? 'champion' : strategy.includes('_xpe') ? 'external_mcs' : 'both',
    compactOpponentTemperature,
    compactOpponentFuture: !strategy.includes('_cf'),
    compactOpponentContextual: strategy.includes('_c3') ? 2 : strategy.includes('_cx'),
    compactHistoryAttempts: compactHistoryMatch ? Number(compactHistoryMatch[1]) : 0,
    compactOpponentTarget: strategy.includes('_pc') ? 'champion' : strategy.includes('_pe') ? 'external_mcs' : 'both',
    compactOpponentVersion2: strategy.includes('_c2'),
    compactStratifiedActions: strategy.includes('_la'),
    deterministicTempo: strategy.includes('_kt'),
    stratified: strategy.includes('_ls'),
    balancedStratified: strategy.includes('_lb'),
    rotateStratifiedPhases: strategy.includes('_lr'),
    handPosterior: Boolean(handPosteriorMatch) && currentTurn >= Number(handPosteriorMatch[1] || 0),
    handPosteriorFraction: handPosteriorFractionMatch ? Number(handPosteriorFractionMatch[1]) / 100 : 1,
    deterministicKnownCautious: strategy.includes('_kc'),
    deterministicKnownRandom: strategy.includes('_kr'),
    seededCautiousExploration: seededCautiousMatch ? Number(seededCautiousMatch[1] ?? 11) / 100 : null,
    deterministicKnownCautiousRoot: strategy.includes('_ckr'),
    opponentProfile: exactProfileActive ? `known-${exactProfileMatch[1]}-${exactProfileMatch[2]}` : knownArena ? 'known-c-g' : 'mixed',
  })
  const residualMatch = strategy.match(/_rc(\d+)?(?:_|$)/)
  const residualStartMatch = strategy.match(/_rds([0-9])(?:_|$)/)
  const residualEndMatch = strategy.match(/_rdy([1-9])(?:_|$)/)
  const residualActive = (!residualStartMatch || currentTurn >= Number(residualStartMatch[1]))
    && (!residualEndMatch || currentTurn < Number(residualEndMatch[1]))
  if (residualMatch && residualActive) {
    const blend = residualMatch[1] == null ? Number(searchResidualModel.blend ?? 1) : Number(residualMatch[1]) / 100
    searchResult = applySearchResidual(observation, searchResult, blend)
  }
  if (strategy.includes('_sg')) {
    const counterfactualEvaluations = evaluateCounterfactualCards(observation)
    const counterfactualCard = counterfactualEvaluations[0].card
    const searchMargin = searchResult.evaluations.length > 1
      ? searchResult.evaluations[1].utility - searchResult.evaluations[0].utility
      : 10
    const counterfactualMargin = counterfactualEvaluations.length > 1
      ? (counterfactualEvaluations[0].logit ?? -counterfactualEvaluations[0].utility) - (counterfactualEvaluations[1].logit ?? -counterfactualEvaluations[1].utility)
      : 10
    const gateFeatures = [
      ...encodePolicyV2State(observation),
      searchResult.card / 54,
      counterfactualCard / 54,
      Math.max(-2, Math.min(2, searchMargin / 5)),
      Math.max(-2, Math.min(2, counterfactualMargin / 5)),
      Number(searchResult.card === counterfactualCard),
      (counterfactualCard - searchResult.card) / 54,
    ]
    if (forwardLayers(gateFeatures, searchCounterfactualGateModel.layers)[0] > 0) {
      return { ...searchResult, card: counterfactualCard, searchGate: true }
    }
  }
  return searchResult
}

export function chooseCard(observation, strategy = 'champion') {
  if (!observation.hand?.length) throw new Error('AI 没有可出的手牌')
  if ((strategy === 'neural_hybrid' || strategy === 'neural_hybrid_zxcg') && observation.deckSize === 54 && observation.playerCount === 5) {
    return policyV2HybridChoice(observation, FINAL_ADAPTIVE_STRATEGY)
  }
  if (strategy === 'policy_v2') return choosePolicyV2Card(observation, policyV2Model)
  if (strategy.startsWith('neural_hybrid_v2')) return policyV2HybridChoice(observation, strategy)
  if (strategy === 'neural_rl') return chooseNeuralCard(observation, outcomeModelFor(observation))
  const rng = mulberry32(observation.seed || 0x6e696d6d)
  if (strategy === 'neural') return chooseNeuralCard(observation, neuralModelFor(observation))
  if (strategy.startsWith('neural_hybrid')) {
    const classicMode = observation.deckSize > observation.playerCount * 10 + 4
    const twoPlayer = strategy === 'neural_hybrid' && observation.playerCount === 2
    const threePlayer = strategy === 'neural_hybrid' && observation.playerCount === 3
    const threePlayerClassic = threePlayer && classicMode
    const fivePlayerClassic = strategy === 'neural_hybrid' && observation.playerCount === 5 && classicMode
    const sixPlayerClassic = strategy === 'neural_hybrid' && observation.playerCount === 6 && classicMode
    const weightMatch = strategy.match(/_p(\d+)/)
    const gateMatch = strategy.match(/_g(\d+)/)
    const candidateMatch = strategy.match(/_k(\d+)/)
    const rankMatch = strategy.match(/_r(\d+)/)
    const riskMatch = strategy.match(/_v(\d+)/)
    const relativeMatch = strategy.match(/_d(\d+)/)
    const confidenceMatch = strategy.match(/_c(\d+)/)
    const openingMatch = strategy.match(/_o(\d+)/)
    const priorWeight = weightMatch ? Number(weightMatch[1]) / 100 : sixPlayerClassic || fivePlayerClassic ? 0.15 : twoPlayer && classicMode ? 0.05 : threePlayerClassic ? 0.05 : 0.10
    const classicNeuralThreshold = gateMatch ? Number(gateMatch[1]) : 11
    const candidateFraction = candidateMatch ? Number(candidateMatch[1]) / 100 : null
    const useClassicBook = !strategy.includes('_b0') && !twoPlayer && !threePlayerClassic
    const mcsProfile = strategy.includes('_mcs') || twoPlayer || threePlayer
    const scoring = {
      rankWeight: rankMatch ? Number(rankMatch[1]) / 100 : mcsProfile ? 0 : undefined,
      relativeWeight: relativeMatch ? Number(relativeMatch[1]) / 100 : mcsProfile ? 0 : undefined,
      riskScale: riskMatch ? Number(riskMatch[1]) / 100 : mcsProfile ? 0 : undefined,
      focalStyle: strategy.includes('_fb') ? 'bestResponse' : strategy.includes('_fn') ? 'neural' : strategy.includes('_fg') ? 'greedy' : strategy.includes('_ft') ? 'tempo' : mcsProfile ? 'random' : undefined,
    }
    const confidenceThreshold = confidenceMatch ? Number(confidenceMatch[1]) / 100 : fivePlayerClassic ? 0.50 : 0
    const ensemble = strategy.startsWith('neural_hybrid_ensemble')
    const specialist = strategy.startsWith('neural_hybrid_spec')
    const selectedModel = specialist ? adaptiveFiveModel : strategy.startsWith('neural_hybrid_rl') || ensemble ? outcomeModelFor(observation) : neuralModelFor(observation)
    const arenaMatch = strategy.match(/_a([1-5])(?:_|$)/)
    const exactProfileMatch = strategy.match(/_zx([cgtr])([cgtr])(?:_|$)/)
    const adaptiveArenaProfile = strategy === 'neural_hybrid' && !classicMode && observation.opponentStrategies?.includes('champion') && observation.opponentStrategies?.includes('external_mcs')
    const opponentProfile = exactProfileMatch ? `known-${exactProfileMatch[1]}-${exactProfileMatch[2]}` : adaptiveArenaProfile ? 'known-c-g' : strategy.includes('_z1') ? 'known1' : strategy.includes('_z2') ? 'known2' : arenaMatch ? `arena${arenaMatch[1]}` : mcsProfile ? 'random' : 'mixed'
    const minimumCandidates = strategy.includes('_n2') ? 2 : 3
    const historyMatch = strategy.match(/_bh(\d+)?(?:_|$)/)
    const historyStartMatch = strategy.match(/_bt(\d+)/)
    const historyStart = historyStartMatch ? Number(historyStartMatch[1]) : 1
    const historyAttempts = historyMatch && (observation.history?.length || 0) >= historyStart ? Number(historyMatch[1] || 4) : 0
    const historyMode = strategy.includes('_hs') ? 'seededCautious' : strategy.includes('_hc') ? 'cautious' : strategy.includes('_hr') ? 'rank' : 'exact'
    const opponentPredictMatch = strategy.match(/_xp(\d+)/)
    const opponentPredictSamples = opponentPredictMatch ? Math.max(1, Number(opponentPredictMatch[1])) : 0
    return neuralHybridChoice(observation, priorWeight, classicNeuralThreshold, candidateFraction, useClassicBook, scoring, opponentProfile, confidenceThreshold, selectedModel, ensemble ? neuralModelFor(observation) : null, openingMatch ? Number(openingMatch[1]) : 11, strategy.includes('_sh'), minimumCandidates, !strategy.includes('_h0'), historyAttempts, historyMode, opponentPredictSamples)
  }
  if (strategy === 'external_mcs') return chooseExternalMcsCard(observation)
  if (strategy === 'random') {
    const card = observation.hand[Math.floor(rng() * observation.hand.length)]
    return { card, evaluations: [{ card, utility: 0, risk: 0, disasterRate: 0 }] }
  }
  if (strategy === 'greedy') {
    const card = greedyChoice(observation.hand, observation.rows)
    return { card, evaluations: [{ card, utility: 0, risk: 0, disasterRate: 0 }] }
  }
  if (strategy === 'cautious') {
    const evaluations = rankedChoices(observation.hand, observation.rows, observation).map((entry) => ({ ...entry, utility: entry.cost, risk: 0, disasterRate: 0 }))
    return { card: evaluations[0].card, evaluations }
  }
  // With a large undealt remainder, early deep determinizations are extremely
  // diffuse. Use the calibrated density model for the opening, then switch to
  // information-set search as public evidence accumulates.
  if (strategy === 'champion' && observation.deckSize > observation.playerCount * 10 + 4 && observation.hand.length >= 8) {
    const evaluations = rankedChoices(observation.hand, observation.rows, observation).map((entry) => ({ ...entry, utility: entry.cost, risk: 0, disasterRate: 0 }))
    return { card: evaluations[0].card, evaluations, openingBook: true }
  }
  return searchChoice(observation, strategy !== 'legacy')
}

export function playArenaGame({ playerCount = 5, deckMode = 'adaptive', strategies, seed = 1, championSamples = 24, legacySamples = 12 }) {
  const deal = createGame(playerCount, deckMode, mulberry32(seed))
  const hands = deal.hands.map((hand) => [...hand])
  const scores = Array(playerCount).fill(0)
  let rows = cloneRows(deal.rows)
  const seenCards = [...deal.initialCards]
  const publicHistory = []
  const resolvedStrategies = [...strategies]
  if (deckMode === 'adaptive' && playerCount === 5) {
    resolvedStrategies.forEach((strategy, player) => {
      if (!strategy.includes('_pg')) return
      resolvedStrategies[player] = choosePortfolioStrategy({
        rows: cloneRows(rows),
        hand: [...hands[player]],
        seenCards: [...seenCards],
        deckSize: deal.deckSize,
        playerCount,
        playerSeat: player,
        scores: Array(playerCount).fill(0),
        playedCards: Array.from({ length: playerCount }, () => []),
      })
    })
  }

  for (let turn = 0; turn < 10; turn += 1) {
    const actions = hands.map((hand, player) => {
      const strategy = resolvedStrategies[player]
      const sampleOverride = strategy.match(/_s(\d+)/)
      const samples = sampleOverride ? Number(sampleOverride[1]) : strategy === 'champion' || strategy.startsWith('neural_hybrid') ? championSamples : legacySamples
      const relativePlayers = Array.from({ length: playerCount }, (_, offset) => (player + offset) % playerCount)
      const perspectiveScores = relativePlayers.map((seat) => scores[seat])
      const opponentStrategies = relativePlayers.slice(1).map((seat) => resolvedStrategies[seat])
      const history = publicHistory.map((round) => ({
        rows: cloneRows(round.rows),
        seenCards: [...round.seenCards],
        ownCard: round.cards[player],
        opponentCards: relativePlayers.slice(1).map((seat) => round.cards[seat]),
      }))
      const playedCards = relativePlayers.map((seat) => publicHistory.map((round) => round.cards[seat]))
      return chooseCard({ rows, hand, seenCards, deckSize: deal.deckSize, playerCount, playerSeat: player, scores: perspectiveScores, opponentStrategies, history, playedCards, samples, seed: seed + turn * 104729 + player * 8191 }, strategy).card
    })
    publicHistory.push({ rows: cloneRows(rows), seenCards: [...seenCards], cards: [...actions] })
    actions.forEach((card, player) => removeCard(hands[player], card))
    const result = playCards(rows, actions)
    rows = result.rows
    result.penalties.forEach((penalty, player) => { scores[player] += penalty })
    seenCards.push(...actions)
  }
  return { scores, strategies: resolvedStrategies }
}
