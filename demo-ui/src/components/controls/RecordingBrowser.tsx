import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Folder, FileAudio, Clock, Activity, ChevronRight, RefreshCw, Search, X } from 'lucide-react'

interface Recording {
  path: string
  filename: string
  duration_seconds: number
  num_samples: number
  num_gestures: number
  gesture_counts: Record<string, number>
}

interface RecordingBrowserProps {
  onSelect: (recording: Recording) => void
  selectedPath: string | null
  isOpen: boolean
  onClose: () => void
}

export default function RecordingBrowser({
  onSelect,
  selectedPath,
  isOpen,
  onClose
}: RecordingBrowserProps) {
  const [recordings, setRecordings] = useState<Recording[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [searchQuery, setSearchQuery] = useState('')
  const [modelLoaded, setModelLoaded] = useState(false)

  const fetchRecordings = async () => {
    setLoading(true)
    setError(null)
    try {
      const response = await fetch('/api/recordings')
      if (!response.ok) throw new Error('Failed to fetch recordings')
      const data = await response.json()
      setRecordings(data.recordings || [])
      setModelLoaded(data.model_loaded || false)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    if (isOpen) {
      fetchRecordings()
    }
  }, [isOpen])

  const filteredRecordings = recordings.filter(rec =>
    rec.filename.toLowerCase().includes(searchQuery.toLowerCase())
  )

  const formatDuration = (seconds: number) => {
    const mins = Math.floor(seconds / 60)
    const secs = Math.floor(seconds % 60)
    return `${mins}:${secs.toString().padStart(2, '0')}`
  }

  if (!isOpen) return null

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 bg-black/30 backdrop-blur-sm z-50 flex items-center justify-center p-4"
        onClick={onClose}
      >
        <motion.div
          initial={{ opacity: 0, scale: 0.95, y: 20 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          exit={{ opacity: 0, scale: 0.95, y: 20 }}
          className="bg-white rounded-2xl shadow-2xl w-full max-w-2xl max-h-[80vh] overflow-hidden"
          onClick={e => e.stopPropagation()}
        >
          {/* Header */}
          <div className="p-6 border-b border-gray-100">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-accent-primary to-accent-secondary flex items-center justify-center">
                  <Folder className="w-5 h-5 text-white" />
                </div>
                <div>
                  <h2 className="text-xl font-bold text-gray-800">Select Recording</h2>
                  <p className="text-sm text-gray-500">
                    {recordings.length} recordings found
                    {modelLoaded && (
                      <span className="ml-2 text-emerald-600">• Model loaded</span>
                    )}
                  </p>
                </div>
              </div>
              <button
                onClick={onClose}
                className="p-2 rounded-lg hover:bg-gray-100 transition-colors"
              >
                <X className="w-5 h-5 text-gray-400" />
              </button>
            </div>

            {/* Search */}
            <div className="relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
              <input
                type="text"
                placeholder="Search recordings..."
                value={searchQuery}
                onChange={e => setSearchQuery(e.target.value)}
                className="w-full pl-10 pr-4 py-2.5 bg-gray-50 border border-gray-200 rounded-xl text-sm focus:outline-none focus:ring-2 focus:ring-accent-primary/20 focus:border-accent-primary"
              />
            </div>
          </div>

          {/* Content */}
          <div className="p-4 overflow-y-auto max-h-[50vh]">
            {loading ? (
              <div className="flex flex-col items-center justify-center py-12">
                <RefreshCw className="w-8 h-8 text-accent-primary animate-spin mb-3" />
                <p className="text-gray-500">Loading recordings...</p>
              </div>
            ) : error ? (
              <div className="flex flex-col items-center justify-center py-12">
                <div className="w-12 h-12 rounded-full bg-red-100 flex items-center justify-center mb-3">
                  <X className="w-6 h-6 text-red-500" />
                </div>
                <p className="text-red-600 font-medium">{error}</p>
                <button
                  onClick={fetchRecordings}
                  className="mt-3 text-sm text-accent-primary hover:underline"
                >
                  Try again
                </button>
              </div>
            ) : filteredRecordings.length === 0 ? (
              <div className="flex flex-col items-center justify-center py-12">
                <FileAudio className="w-12 h-12 text-gray-300 mb-3" />
                <p className="text-gray-500">No recordings found</p>
                <p className="text-sm text-gray-400 mt-1">
                  Place HDF5 files in the data directory
                </p>
              </div>
            ) : (
              <div className="space-y-2">
                {filteredRecordings.map((recording, index) => (
                  <motion.button
                    key={recording.path}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: index * 0.05 }}
                    onClick={() => onSelect(recording)}
                    className={`w-full p-4 rounded-xl border-2 text-left transition-all ${
                      selectedPath === recording.path
                        ? 'border-accent-primary bg-indigo-50'
                        : 'border-gray-100 hover:border-gray-200 hover:bg-gray-50'
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${
                          selectedPath === recording.path
                            ? 'bg-accent-primary text-white'
                            : 'bg-gray-100 text-gray-500'
                        }`}>
                          <FileAudio className="w-5 h-5" />
                        </div>
                        <div>
                          <p className="font-medium text-gray-800 truncate max-w-[300px]">
                            {recording.filename}
                          </p>
                          <div className="flex items-center gap-4 mt-1 text-sm text-gray-500">
                            <span className="flex items-center gap-1">
                              <Clock className="w-3.5 h-3.5" />
                              {formatDuration(recording.duration_seconds)}
                            </span>
                            <span className="flex items-center gap-1">
                              <Activity className="w-3.5 h-3.5" />
                              {recording.num_gestures} gestures
                            </span>
                          </div>
                        </div>
                      </div>
                      <ChevronRight className={`w-5 h-5 transition-colors ${
                        selectedPath === recording.path
                          ? 'text-accent-primary'
                          : 'text-gray-300'
                      }`} />
                    </div>

                    {/* Gesture breakdown */}
                    {recording.num_gestures > 0 && (
                      <div className="mt-3 flex flex-wrap gap-1.5">
                        {Object.entries(recording.gesture_counts)
                          .filter(([_, count]) => count > 0)
                          .map(([gesture, count]) => (
                            <span
                              key={gesture}
                              className="px-2 py-0.5 bg-gray-100 text-gray-600 text-xs rounded-full"
                            >
                              {gesture.replace('_', ' ')}: {count}
                            </span>
                          ))}
                      </div>
                    )}
                  </motion.button>
                ))}
              </div>
            )}
          </div>

          {/* Footer */}
          <div className="p-4 border-t border-gray-100 bg-gray-50">
            <div className="flex items-center justify-between">
              <button
                onClick={fetchRecordings}
                className="flex items-center gap-2 px-4 py-2 text-sm text-gray-600 hover:text-gray-800 transition-colors"
              >
                <RefreshCw className="w-4 h-4" />
                Refresh
              </button>
              <p className="text-xs text-gray-400">
                Supported formats: .hdf5, .h5
              </p>
            </div>
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  )
}
