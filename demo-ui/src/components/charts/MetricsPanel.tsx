import { motion } from 'framer-motion'
import { Zap, Target, Gauge, Activity } from 'lucide-react'

interface MetricsPanelProps {
  metrics: {
    latency: number
    accuracy: number
    modelConfidence: number
    samplesProcessed: number
  }
}

interface MetricCardProps {
  icon: React.ReactNode
  label: string
  value: string
  subValue?: string
  color: string
  progress?: number
}

function MetricCard({ icon, label, value, subValue, color, progress }: MetricCardProps) {
  return (
    <motion.div
      className="metric-card glass-panel rounded-xl p-5"
      whileHover={{ y: -2 }}
    >
      <div className="flex items-start justify-between mb-3">
        <div
          className="w-10 h-10 rounded-lg flex items-center justify-center"
          style={{ backgroundColor: `${color}15` }}
        >
          <div style={{ color }}>{icon}</div>
        </div>
        <span className="text-xs font-medium text-gray-400 uppercase tracking-wider">
          {label}
        </span>
      </div>

      <div className="space-y-2">
        <div className="flex items-baseline gap-1">
          <span
            className="text-3xl font-bold font-display"
            style={{ color }}
          >
            {value}
          </span>
          {subValue && (
            <span className="text-sm text-gray-400">{subValue}</span>
          )}
        </div>

        {progress !== undefined && (
          <div className="h-2 bg-gray-100 rounded-full overflow-hidden">
            <motion.div
              className="h-full rounded-full"
              style={{ backgroundColor: color }}
              initial={{ width: 0 }}
              animate={{ width: `${Math.min(progress, 100)}%` }}
              transition={{ duration: 0.5, ease: 'easeOut' }}
            />
          </div>
        )}
      </div>
    </motion.div>
  )
}

export default function MetricsPanel({ metrics }: MetricsPanelProps) {
  return (
    <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
      <MetricCard
        icon={<Zap className="w-5 h-5" />}
        label="Latency"
        value={metrics.latency > 0 ? metrics.latency.toFixed(1) : '—'}
        subValue="ms"
        color="#06B6D4"
        progress={metrics.latency > 0 ? (10 - metrics.latency) * 10 : 0}
      />

      <MetricCard
        icon={<Target className="w-5 h-5" />}
        label="Accuracy"
        value={metrics.accuracy > 0 ? metrics.accuracy.toFixed(1) : '—'}
        subValue="%"
        color="#10B981"
        progress={metrics.accuracy}
      />

      <MetricCard
        icon={<Gauge className="w-5 h-5" />}
        label="Confidence"
        value={metrics.modelConfidence > 0 ? metrics.modelConfidence.toFixed(1) : '—'}
        subValue="%"
        color="#6366F1"
        progress={metrics.modelConfidence}
      />

      <MetricCard
        icon={<Activity className="w-5 h-5" />}
        label="Throughput"
        value={metrics.samplesProcessed > 0 ? (metrics.samplesProcessed / 5).toFixed(0) : '—'}
        subValue="Hz"
        color="#8B5CF6"
        progress={Math.min((metrics.samplesProcessed / 5) / 2, 100)}
      />
    </div>
  )
}
