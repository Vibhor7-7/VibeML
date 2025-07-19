'use client'

import { useState, useEffect } from 'react'
import { BarChart3, Download, Eye, RefreshCw, TrendingUp } from 'lucide-react'

interface Model {
  model_id: number
  algorithm: string
  dataset: string
  accuracy: number | null
  target_column: string
  created_at: string
  model_size_mb: number
}

interface ModelListProps {
  onModelSelect: (modelId: number) => void
  selectedModelId?: number | null
}

export default function ModelList({ onModelSelect, selectedModelId }: ModelListProps) {
  const [models, setModels] = useState<Model[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchModels = async () => {
    setLoading(true)
    setError(null)
    
    try {
      const response = await fetch('/api/predict/models')
      
      if (response.ok) {
        const data = await response.json()
        setModels(data.available_models || [])
      } else {
        throw new Error('Failed to fetch models')
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load models')
      // Show mock data for demo
      setModels([
        {
          model_id: 1,
          algorithm: 'random_forest',
          dataset: 'iris_dataset',
          accuracy: 0.967,
          target_column: 'species',
          created_at: '2025-07-19T10:30:00Z',
          model_size_mb: 2.1
        },
        {
          model_id: 2,
          algorithm: 'logistic_regression',
          dataset: 'titanic_dataset',
          accuracy: 0.823,
          target_column: 'survived',
          created_at: '2025-07-19T09:15:00Z',
          model_size_mb: 0.8
        }
      ])
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchModels()
  }, [])

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    })
  }

  const getAlgorithmColor = (algorithm: string) => {
    const colors: Record<string, string> = {
      'random_forest': 'bg-green-100 text-green-800',
      'logistic_regression': 'bg-blue-100 text-blue-800',
      'svm': 'bg-purple-100 text-purple-800',
      'knn': 'bg-orange-100 text-orange-800',
      'decision_tree': 'bg-yellow-100 text-yellow-800',
      'neural_network': 'bg-red-100 text-red-800'
    }
    return colors[algorithm] || 'bg-gray-100 text-gray-800'
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="text-center">
          <div className="w-8 h-8 border-2 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto mb-4" />
          <p className="text-gray-600">Loading available models...</p>
        </div>
      </div>
    )
  }

  if (error && models.length === 0) {
    return (
      <div className="text-center py-12">
        <BarChart3 className="w-12 h-12 mx-auto mb-4 text-gray-400" />
        <p className="text-gray-600 mb-4">Failed to load models</p>
        <button
          onClick={fetchModels}
          className="flex items-center gap-2 mx-auto px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
        >
          <RefreshCw className="w-4 h-4" />
          Retry
        </button>
      </div>
    )
  }

  if (models.length === 0) {
    return (
      <div className="text-center py-12">
        <BarChart3 className="w-12 h-12 mx-auto mb-4 text-gray-400" />
        <p className="text-gray-600 mb-2">No trained models available</p>
        <p className="text-sm text-gray-500">Train some models first to see them here</p>
      </div>
    )
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold text-gray-900">
          Available Models ({models.length})
        </h3>
        <button
          onClick={fetchModels}
          className="flex items-center gap-2 px-3 py-2 text-sm text-gray-600 hover:text-gray-800"
        >
          <RefreshCw className="w-4 h-4" />
          Refresh
        </button>
      </div>

      <div className="grid gap-4">
        {models.map((model) => (
          <div
            key={model.model_id}
            className={`border rounded-lg p-4 hover:shadow-md transition-all cursor-pointer ${
              selectedModelId === model.model_id
                ? 'border-blue-500 bg-blue-50 shadow-md'
                : 'border-gray-200 hover:border-gray-300'
            }`}
            onClick={() => onModelSelect(model.model_id)}
          >
            <div className="flex items-start justify-between">
              <div className="flex-1">
                <div className="flex items-center gap-2 mb-2">
                  <span className={`px-2 py-1 text-xs font-medium rounded-full ${getAlgorithmColor(model.algorithm)}`}>
                    {model.algorithm.replace('_', ' ').toUpperCase()}
                  </span>
                  <span className="text-sm text-gray-500">
                    Model #{model.model_id}
                  </span>
                </div>

                <h4 className="font-medium text-gray-900 mb-1">
                  {model.dataset.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
                </h4>

                <div className="flex items-center gap-4 text-sm text-gray-600 mb-3">
                  <span>Target: {model.target_column}</span>
                  <span>Size: {model.model_size_mb}MB</span>
                  <span>Created: {formatDate(model.created_at)}</span>
                </div>

                {model.accuracy && (
                  <div className="flex items-center gap-2">
                    <TrendingUp className="w-4 h-4 text-green-600" />
                    <span className="text-sm font-medium text-green-600">
                      Accuracy: {(model.accuracy * 100).toFixed(1)}%
                    </span>
                  </div>
                )}
              </div>

              <div className="flex gap-2 ml-4">
                <button
                  className="p-2 text-gray-400 hover:text-blue-600"
                  onClick={(e) => {
                    e.stopPropagation()
                    // View model details
                    window.open(`/api/predict/models/${model.model_id}/info`, '_blank')
                  }}
                  title="View Details"
                >
                  <Eye className="w-4 h-4" />
                </button>
                <button
                  className="p-2 text-gray-400 hover:text-green-600"
                  onClick={(e) => {
                    e.stopPropagation()
                    // Download model
                    window.open(`/api/export/model/${model.model_id}`, '_blank')
                  }}
                  title="Download Model"
                >
                  <Download className="w-4 h-4" />
                </button>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
