import { motion } from 'framer-motion'
import { Check, X, Target, TrendingUp } from 'lucide-react'
import { GESTURE_LABELS, GestureType } from '../../types/gestures'

interface GroundTruth {
  time: number
  gesture: string
  gesture_id: number
}

interface PredictionComparisonProps {
  predictions: number[]  // 9 probabilities
  detectedGesture: number | null
  groundTruth: GroundTruth | null
  confidence: number
  isCorrect: boolean
}

const GESTURE_COLORS: Record<number, string> = {
  0: '#6366F1', // index_press
  1: '#818CF8', // index_release
  2: '#06B6D4', // middle_press
  3: '#22D3EE', // middle_release
  4: '#8B5CF6', // thumb_click
  5: '#F59E0B', // thumb_down
  6: '#14B8A6', // thumb_in
  7: '#EC4899', // thumb_out
  8: '#10B981', // thumb_up
}

export default function PredictionComparison({
  predictions,
  detectedGesture,
  groundTruth,
  confidence,
  isCorrect
}: PredictionComparisonProps) {
  // Sort predictions by probability for display
  const sortedPredictions = predictions
    .map((prob, idx) => ({ idx, prob, label: GESTURE_LABELS[idx as GestureType] }))
    .sort((a, b) => b.prob - a.prob)
    .slice(0, 5) // Show top 5

  return (
    <div className="glass-panel rounded-2xl p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-gray-800">Prediction Analysis</h3>
        {groundTruth && (
          <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-sm font-medium ${
            isCorrect
              ? 'bg-emerald-100 text-emerald-700'
              : 'bg-red-100 text-red-700'
          }`}>
            {isCorrect ? (
              <>
                <Check className="w-4 h-4" />
                Correct
              </>
            ) : (
              <>
                <X className="w-4 h-4" />
                Incorrect
              </>
            )}
          </div>
        )}
      </div>

      <div className="grid grid-cols-2 gap-6">
        {/* Left: Ground Truth */}
        <div>
          <div className="flex items-center gap-2 mb-3">
            <Target className="w-4 h-4 text-gray-400" />
            <span className="text-sm font-medium text-gray-500">Ground Truth</span>
          </div>

          {groundTruth ? (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="p-4 rounded-xl border-2 border-dashed border-gray-200 bg-gray-50"
            >
              <div className="flex items-center gap-3">
                <div
                  className="w-12 h-12 rounded-xl flex items-center justify-center text-white font-bold"
                  style={{ backgroundColor: GESTURE_COLORS[groundTruth.gesture_id] }}
                >
                  {groundTruth.gesture_id}
                </div>
                <div>
                  <p className="font-semibold text-gray-800 capitalize">
                    {groundTruth.gesture.replace('_', ' ')}
                  </p>
                  <p className="text-sm text-gray-500">
                    @ {groundTruth.time.toFixed(2)}s
                  </p>
                </div>
              </div>
            </motion.div>
          ) : (
            <div className="p-4 rounded-xl border-2 border-dashed border-gray-200 bg-gray-50 text-center">
              <p className="text-gray-400">No gesture at this time</p>
            </div>
          )}
        </div>

        {/* Right: Prediction */}
        <div>
          <div className="flex items-center gap-2 mb-3">
            <TrendingUp className="w-4 h-4 text-gray-400" />
            <span className="text-sm font-medium text-gray-500">Model Prediction</span>
          </div>

          {detectedGesture !== null ? (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className={`p-4 rounded-xl border-2 ${
                isCorrect
                  ? 'border-emerald-200 bg-emerald-50'
                  : groundTruth
                    ? 'border-red-200 bg-red-50'
                    : 'border-indigo-200 bg-indigo-50'
              }`}
            >
              <div className="flex items-center gap-3">
                <div
                  className="w-12 h-12 rounded-xl flex items-center justify-center text-white font-bold"
                  style={{ backgroundColor: GESTURE_COLORS[detectedGesture] }}
                >
                  {detectedGesture}
                </div>
                <div>
                  <p className="font-semibold text-gray-800">
                    {GESTURE_LABELS[detectedGesture as GestureType]}
                  </p>
                  <p className="text-sm text-gray-500">
                    {confidence.toFixed(1)}% confidence
                  </p>
                </div>
              </div>
            </motion.div>
          ) : (
            <div className="p-4 rounded-xl border-2 border-dashed border-gray-200 bg-gray-50 text-center">
              <p className="text-gray-400">No detection (threshold: 50%)</p>
            </div>
          )}
        </div>
      </div>

      {/* Probability Distribution */}
      <div className="mt-6">
        <h4 className="text-sm font-medium text-gray-500 mb-3">Probability Distribution (Top 5)</h4>
        <div className="space-y-2">
          {sortedPredictions.map(({ idx, prob, label }) => (
            <div key={idx} className="flex items-center gap-3">
              <div className="w-24 text-xs text-gray-600 truncate">{label}</div>
              <div className="flex-1 h-6 bg-gray-100 rounded-full overflow-hidden">
                <motion.div
                  className="h-full rounded-full"
                  style={{ backgroundColor: GESTURE_COLORS[idx] }}
                  initial={{ width: 0 }}
                  animate={{ width: `${prob * 100}%` }}
                  transition={{ duration: 0.3 }}
                />
              </div>
              <div className="w-14 text-xs font-mono text-gray-500 text-right">
                {(prob * 100).toFixed(1)}%
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
