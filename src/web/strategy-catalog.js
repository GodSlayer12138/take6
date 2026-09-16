import { modelFor } from './ai-router.js'

export const POOLS = {
  2: ['strongest', 'external_mcs'],
  3: ['strongest', 'champion', 'external_mcs'],
  4: ['strongest', 'champion', 'external_mcs', 'random'],
}

export function strategiesFor(config) {
  const model = modelFor(config)
  return {
    strongest: { id: 'strongest', name: model.label, description: model.remote ? '三四人 104 张正式验收版本' : model.id === 'v6' ? '二人 104 张专项策略' : '当前规则使用历史浏览器策略', color: '#f2c14e' },
    champion: { id: 'champion', name: '牧场冠军 v2', description: '历史信息集搜索对照', color: '#ef9c5d' },
    external_mcs: { id: 'external_mcs', name: '公开 MCS · 适配', description: '公开蒙特卡洛搜索对照', color: '#d8b15f' },
    random: { id: 'random', name: '随机基线', description: '从合法手牌随机出牌', color: '#8b9a95' },
  }
}
