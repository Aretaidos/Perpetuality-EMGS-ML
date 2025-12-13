import { motion } from 'framer-motion'
import { Target, Zap, Gauge, BarChart3, CheckCircle, XCircle } from 'lucide-react'

interface SessionMetricsProps {
  metrics: {
    total_predictions: number
    correct_predictions: number
    accuracy: number
    avg_latency_ms: number
    avg_confidence: number
    gesture_accuracies: Record<string, number>
  } | null
  isComplete: boolean
}

export default function SessionMetrics({ metrics, isComplete }: SessionMetricsProps) {
  if (!metrics) {
    return (
      <div className="glass-panel rounded-2xl p-6">
        <h3 className="text-lg font-semibold text-gray-800 mb-4">Session Metrics</h3>
        <div className="flex items-center justify-center py-8 text-gray-400">
          <p>Metrics will appear after playback</p>
        </div>
      </div>
    )
  }

  const gesturesToShow = Object.entries(metrics.gesture_accuracies)
    .filter(([_, acc]) => acc > 0)
    .sort((a, b) => b[1] - a[1])

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="glass-panel rounded-2xl p-6"
    >
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-lg font-semibold text-gray-800">Session Metrics</h3>
        {isComplete && (
          <span className="px-3 py-1 bg-emerald-100 text-emerald-700 text-sm font-medium rounded-full">
            Complete
          </span>
        )}
      </div>

      {/* Main metrics grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        {/* Accuracy */}
        <div className="p-4 bg-gradient-to-br from-emerald-50 to-emerald-100/50 rounded-xl">
          <div className="flex items-center gap-2 mb-2">
            <Target className="w-4 h-4 text-emerald-600" />
            <span className="text-xs font-medium text-emerald-600">Accuracy</span>
          </div>
          <p className="text-2xl font-bold text-emerald-700">
            {metrics.accuracy.toFixed(1)}%
          </p>
        </div>

        {/* Latency */}
        <div className="p-4 bg-gradient-to-br from-cyan-50 to-cyan-100/50 rounded-xl">
          <div className="flex items-center gap-2 mb-2">
            <Zap className="w-4 h-4 text-cyan-600" />
            <span className="text-xs font-medium text-cyan-600">Avg Latency</span>
          </div>
          <p className="text-2xl font-bold text-cyan-700">
            {metrics.avg_latency_ms.toFixed(1)}ms
          </p>
        </div>

        {/* Confidence */}
        <div className="p-4 bg-gradient-to-br from-purple-50 to-purple-100/50 rounded-xl">
          <div className="flex items-center gap-2 mb-2">
            <Gauge className="w-4 h-4 text-purple-600" />
            <span className="text-xs font-medium text-purple-600">Avg Confidence</span>
          </div>
          <p className="text-2xl font-bold text-purple-700">
            {metrics.avg_confidence.toFixed(1)}%
          </p>
        </div>

        {/* Total predictions */}
        <div className="p-4 bg-gradient-to-br from-indigo-50 to-indigo-100/50 rounded-xl">
          <div className="flex items-center gap-2 mb-2">
            <BarChart3 className="w-4 h-4 text-indigo-600" />
            <span className="text-xs font-medium text-indigo-600">Predictions</span>
          </div>
          <p className="text-2xl font-bold text-indigo-700">
            {metrics.total_predictions}
          </p>
        </div>
      </div>

      {/* Correct/Incorrect breakdown */}
      <div className="flex items-center gap-6 mb-6 p-4 bg-gray-50 rounded-xl">
        <div className="flex items-center gap-2">
          <CheckCircle className="w-5 h-5 text-emerald-500" />
          <span className="text-sm text-gray-600">Correct:</span>
          <span className="font-semibold text-emerald-600">{metrics.correct_predictions}</span>
        </div>
        <div className="flex items-center gap-2">
          <XCircle className="w-5 h-5 text-red-400" />
          <span className="text-sm text-gray-600">Incorrect:</span>
          <span className="font-semibold text-red-500">
            {metrics.total_predictions - metrics.correct_predictions}
          </span>
        </div>
        <div className="flex-1 h-3 bg-gray-200 rounded-full overflow-hidden">
          <div
            className="h-full bg-gradient-to-r from-emerald-400 to-emerald-500 rounded-full"
            style={{ width: `${metrics.accuracy}%` }}
          />
        </div>
      </div>

      {/* Per-gesture accuracy */}
      {gesturesToShow.length > 0 && (
        <div>
          <h4 className="text-sm font-medium text-gray-500 mb-3">Per-Gesture Accuracy</h4>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
            {gesturesToShow.map(([gesture, accuracy]) => (
              <div
                key={gesture}
                className="flex items-center justify-between p-2 bg-gray-50 rounded-lg"
              >
                <span className="text-xs text-gray-600 capitalize truncate">
                  {gesture.replace('_', ' ')}
                </span>
                <span className={`text-xs font-semibold ${
                  accuracy >= 90 ? 'text-emerald-600' :
                  accuracy >= 70 ? 'text-amber-600' :
                  'text-red-500'
                }`}>
                  {accuracy.toFixed(0)}%
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </motion.div>
  )
}
