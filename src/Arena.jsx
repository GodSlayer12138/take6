import { useEffect, useRef, useState } from 'react'
import {
  ArrowLeft,
  BarChart3,
  Bot,
  BrainCircuit,
  CheckCircle2,
  ChevronRight,
  CircleStop,
  Crown,
  FlaskConical,
  Gauge,
  Layers3,
  Play,
  RotateCcw,
  ShieldCheck,
  Swords,
  Target,
  Trophy,
  Users,
  Zap,
} from 'lucide-react'
import { POOLS, strategiesFor } from './web/strategy-catalog.js'
import ModelStatus from './web/ModelStatus.jsx'

function StrategyBadge({ strategy, rank, stats, config }) {
  const meta = strategiesFor(config)[strategy]
  return (
    <article className={`arena-strategy-card ${strategy === 'strongest' ? 'champion-card' : ''}`} style={{ '--strategy-color': meta.color }}>
      <div className="strategy-rank">{rank ? `#${rank}` : <Bot size={17} />}</div>
      <div className="strategy-icon">{strategy === 'strongest' ? <Crown /> : strategy === 'champion' ? <BrainCircuit /> : strategy === 'random' ? <FlaskConical /> : <Target />}</div>
      <div className="strategy-copy">
        <div>{meta.name}{strategy === 'strongest' && <span>推荐 AI</span>}</div>
        <small>{meta.description}</small>
      </div>
      {stats ? (
        <div className="strategy-live-stats">
          <b>{(stats.winRate * 100).toFixed(1)}%</b>
          <span>胜率</span>
        </div>
      ) : (
        <div className="strategy-ready"><i /> READY</div>
      )}
    </article>
  )
}

function MetricBar({ value, max, color }) {
  return <span className="metric-bar"><i style={{ width: `${Math.max(4, value / Math.max(1, max) * 100)}%`, background: color }} /></span>
}

export default function Arena({ onBack }) {
  const [config, setConfig] = useState({ games: 100, playerCount: 4, deckMode: 'classic' })
  const [status, setStatus] = useState('idle')
  const [progress, setProgress] = useState(0)
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const workerRef = useRef(null)
  const activePool = POOLS[config.playerCount]
  useEffect(() => () => workerRef.current?.terminate(), [])
  useEffect(() => {
    setResult(null)
    setProgress(0)
    setError('')
    setStatus('idle')
  }, [config.playerCount, config.deckMode, config.games])

  function stopArena() {
    workerRef.current?.terminate()
    workerRef.current = null
    setStatus('idle')
  }

  function runArena() {
    workerRef.current?.terminate()
    const worker = new Worker(new URL('./web/arena.worker.js', import.meta.url), { type: 'module' })
    workerRef.current = worker
    setStatus('running')
    setProgress(0)
    setResult(null)
    setError('')

    worker.onmessage = (event) => {
      if (event.data.type === 'error') {
        setError(event.data.message)
        setStatus('error')
        worker.terminate()
        workerRef.current = null
        return
      }
      setProgress(event.data.completed / event.data.total)
      setResult(event.data)
      if (event.data.type === 'complete') {
        setStatus('complete')
        worker.terminate()
        workerRef.current = null
      }
    }
    worker.onerror = (event) => {
      setError(event.message || '竞技场 Worker 运行失败')
      setStatus('error')
      worker.terminate()
      workerRef.current = null
    }
    worker.postMessage({ ...config, seed: 20260820 })
  }

  const maxWinRate = result ? Math.max(...result.strategies.map((item) => item.winRate)) : 1
  const gpuChampion = result?.strategies.find((item) => item.id === 'strongest')
  const champion = result?.strategies.find((item) => item.id === 'champion')
  const scoreGain = gpuChampion && champion ? (champion.avgScore - gpuChampion.avgScore) / Math.max(0.1, champion.avgScore) : 0

  return (
    <main className="arena-shell">
      <header className="arena-nav">
        <div className="brand-mark small"><span>♉</span><div><b>牛头王</b><small>STRATEGY LAB</small></div></div>
        <div className="arena-nav-title"><Swords size={15} /><span>AI 竞技场</span><i />本机对照实验</div>
        <button type="button" onClick={() => { stopArena(); onBack() }}><ArrowLeft /> 返回牌桌</button>
      </header>

      <section className="arena-content">
        <div className="arena-hero">
          <div className="arena-hero-copy">
            <div className="eyebrow"><Trophy size={15} /> BULLHEAD BENCHMARK</div>
            <h1>让策略自己<span>说话。</span></h1>
            <p>固定随机种子、逐局轮换座位、隐藏信息隔离。比较当前推荐 AI 与历史策略；正式强度结论以独立验收报告为准。</p>
            <div className="champion-tech">
              <span><ShieldCheck /> 公开状态</span><span><BrainCircuit /> 按人数选模型</span><span><Zap /> 异步推演</span>
            </div>
          </div>

          <aside className="arena-control">
            <div className="arena-control-title"><div><span>TOURNAMENT SETUP</span><b>锦标赛配置</b></div><Gauge /></div>
            <label>
              <span><BarChart3 /> 对局数量</span>
              <div className="arena-segments">
                {[20, 50, 100, 200].map((games) => <button type="button" key={games} className={config.games === games ? 'active' : ''} onClick={() => setConfig({ ...config, games })} disabled={status === 'running'}>{games}</button>)}
              </div>
            </label>
            <label>
              <span><Users /> 同场策略</span>
              <div className="arena-segments">
                {[2, 3, 4].map((playerCount) => <button type="button" key={playerCount} className={config.playerCount === playerCount ? 'active' : ''} onClick={() => setConfig({ ...config, playerCount })} disabled={status === 'running'}>{playerCount} 种</button>)}
              </div>
            </label>
            <label>
              <span><Layers3 /> 牌池模式</span>
              <div className="arena-segments two">
                <button type="button" className={config.deckMode === 'adaptive' ? 'active' : ''} onClick={() => setConfig({ ...config, deckMode: 'adaptive' })} disabled={status === 'running'}>10n+4</button>
                <button type="button" className={config.deckMode === 'classic' ? 'active' : ''} onClick={() => setConfig({ ...config, deckMode: 'classic' })} disabled={status === 'running'}>固定 104</button>
              </div>
            </label>

            <ModelStatus config={config} />
            {status === 'running' ? (
              <button className="arena-run stop" type="button" onClick={stopArena}><CircleStop /> 停止比赛 <span>{Math.round(progress * 100)}%</span></button>
            ) : (
              <button className="arena-run" type="button" onClick={runArena}>{status === 'complete' ? <RotateCcw /> : <Play />} {status === 'complete' ? '使用相同种子重赛' : '启动竞技场'} <ChevronRight /></button>
            )}
            {status === 'running' && <div className="arena-progress"><i style={{ width: `${progress * 100}%` }} /></div>}
            {error && <p className="arena-error">{error}</p>}
          </aside>
        </div>

        <section className="arena-roster">
          <div className="arena-section-heading"><div><span>CONTENDERS</span><h2>参赛策略</h2></div><small>{activePool.length} 个策略 · 每局轮换座位</small></div>
          <div className="strategy-card-grid">
            {activePool.map((id) => {
              const stats = result?.strategies.find((item) => item.id === id)
              const rank = result ? result.strategies.findIndex((item) => item.id === id) + 1 : null
              return <StrategyBadge key={id} strategy={id} stats={stats} rank={rank} config={config} />
            })}
          </div>
        </section>

        <section className="arena-results">
          <div className="arena-section-heading">
            <div><span>LIVE RANKING</span><h2>{status === 'complete' ? `${config.games} 局最终排名` : status === 'error' ? `比赛中断 · 已完成 ${result?.completed || 0} 局` : status === 'running' || result ? `已完成 ${result?.completed || 0} / ${config.games} 局` : '等待鸣锣'}</h2></div>
            {status === 'complete' && <div className="verified-badge"><CheckCircle2 /> 基准已完成</div>}
          </div>

          {!result ? (
            <div className="arena-empty-results"><div className="arena-rings"><Trophy /></div><h3>选择赛制，开始批量自博弈</h3><p>所有策略只读取自己的手牌与公共信息；相同随机种子可复现实验。</p></div>
          ) : (
            <>
              <div className="arena-table">
                <div className="arena-table-head"><span>排名 / 策略</span><span>胜率</span><span>平均牛头</span><span>平均名次</span><span>Elo</span></div>
                {result.strategies.map((item, index) => (
                  <div className={`arena-table-row ${item.id === 'strongest' ? 'champion-row' : ''}`} key={item.id}>
                    <div><em>{index + 1}</em><i style={{ background: item.color }} /><span><b>{item.name}</b><small>{item.description}</small></span></div>
                    <div className="win-metric"><b>{(item.winRate * 100).toFixed(1)}%</b><MetricBar value={item.winRate} max={maxWinRate} color={item.color} /><small>本批观察胜率</small></div>
                    <strong>{item.avgScore.toFixed(2)}</strong>
                    <strong>{item.avgRank.toFixed(2)}</strong>
                    <strong className="elo-value">{Math.round(item.elo)}</strong>
                  </div>
                ))}
              </div>

              <div className="arena-insights">
                {champion && <article><span><Target /> 对冠军 v2 平均减分</span><b className={scoreGain >= 0 ? 'positive' : 'negative'}>{scoreGain >= 0 ? '−' : '+'}{Math.abs(scoreGain * 100).toFixed(1)}%</b><small>推荐 AI 相对历史搜索冠军</small></article>}
                {champion && <article><span><Swords /> 推荐 AI vs 冠军 v2</span><b>{`${result.headToHead.champion.gpu} : ${result.headToHead.champion.opponent}`}</b><small>推荐 AI 胜 : v2 胜，另 {result.headToHead.champion.ties} 平</small></article>}
                <article><span><ShieldCheck /> 推荐 AI vs 公开 MCS</span><b>{result ? `${result.headToHead.external.gpu} : ${result.headToHead.external.opponent}` : '—'}</b><small>推荐 AI 胜 : MCS 胜，另 {result?.headToHead.external.ties || 0} 平</small></article>
              </div>
            </>
          )}
        </section>

        <footer className="arena-method">
          <FlaskConical />
          <div><b>实验说明</b><p>每局使用确定性种子发牌，策略逐局轮换座位；并列第一平分胜场。三四人 104 张使用与正式冠军相同的 2,048 个模拟世界，二人 104 张使用 V6。这里的胜率和 Elo 是交互演示统计，对手池与正式验收不同，不能用于宣称新的显著提升。</p></div>
        </footer>
      </section>
    </main>
  )
}
