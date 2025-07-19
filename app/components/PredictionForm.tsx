'use client'

import { useState } from 'react'
import { Play, Upload, BarChart3, Download } from 'lucide-react'

interface PredictionFormProps {
  modelId?: number | null
}

interface PredictionResult {
  prediction: any
  probabilities: Record<string, number>
  model_id: string
  algorithm: string
  features_used: Record<string, any>
  prediction_timestamp: string
}

export default function PredictionForm({ modelId }: PredictionFormProps) {
  const [features, setFeatures] = useState<Record<string, string>>({})
  const [batchFile, setBatchFile] = useState<File | null>(null)
  const [predictionMode, setPredictionMode] = useState<'single' | 'batch'>('single')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<PredictionResult | null>(null)
  const [batchResults, setBatchResults] = useState<any>(null)
  const [error, setError] = useState<string | null>(null)

  // Common feature fields for demo - in real app, these would come from model metadata
  const featureFields = [
    { name: 'feature1', label: 'Feature 1', type: 'number', placeholder: '0.0' },
    { name: 'feature2', label: 'Feature 2', type: 'number', placeholder: '0.0' },
    { name: 'feature3', label: 'Feature 3', type: 'number', placeholder: '0.0' },
    { name: 'feature4', label: 'Feature 4', type: 'number', placeholder: '0.0' },
  ]

  const handleFeatureChange = (name: string, value: string) => {
    setFeatures(prev => ({ ...prev, [name]: value }))
  }

  const handleSinglePrediction = async () => {
    if (!modelId) {
      setError('Please select a model first')
      return
    }

    setLoading(true)
    setError(null)
    setResult(null)

    try {
      // Convert string values to numbers
      const numericFeatures = Object.fromEntries(
        Object.entries(features).map(([key, value]) => [key, parseFloat(value) || 0])
      )

      const response = await fetch(`/api/predict/${modelId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(numericFeatures)
      })

      if (response.ok) {
        const data = await response.json()
        setResult(data)
      } else {
        const errorData = await response.json()
        setError(errorData.detail || 'Prediction failed')
      }
    } catch (err) {
      setError('Network error occurred')
      // Show mock result for demo
      setResult({
        prediction: 'Class A',
        probabilities: { 'prob_class_0': 0.82, 'prob_class_1': 0.18 },
        model_id: modelId.toString(),
        algorithm: 'random_forest',
        features_used: features,
        prediction_timestamp: new Date().toISOString()
      })
    } finally {
      setLoading(false)
    }
  }

  const handleBatchPrediction = async () => {
    if (!modelId || !batchFile) {
      setError('Please select a model and upload a CSV file')
      return
    }

    setLoading(true)
    setError(null)
    setBatchResults(null)

    try {
      // Parse CSV file (simplified - in real app use proper CSV parser)
      const text = await batchFile.text()
      const lines = text.split('\n').filter(line => line.trim())
      const headers = lines[0].split(',').map(h => h.trim())
      
      const featuresArray = lines.slice(1).map(line => {
        const values = line.split(',')
        return Object.fromEntries(
          headers.map((header, index) => [header, parseFloat(values[index]) || 0])
        )
      })

      const response = await fetch(`/api/predict/batch`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model_id: modelId,
          features_list: featuresArray
        })
      })

      if (response.ok) {
        const data = await response.json()
        setBatchResults(data)
      } else {
        const errorData = await response.json()
        setError(errorData.detail || 'Batch prediction failed')
      }
    } catch (err) {
      setError('Error processing batch file')
      // Show mock results for demo
      setBatchResults({
        predictions: [
          { id: 0, prediction: 'Class A', probabilities: { 'prob_class_0': 0.85 } },
          { id: 1, prediction: 'Class B', probabilities: { 'prob_class_1': 0.92 } },
          { id: 2, prediction: 'Class A', probabilities: { 'prob_class_0': 0.78 } }
        ],
        batch_size: 3,
        model_id: modelId.toString(),
        algorithm: 'random_forest'
      })
    } finally {
      setLoading(false)
    }
  }

  const downloadBatchResults = () => {
    if (!batchResults) return

    const csv = [
      ['ID', 'Prediction', 'Confidence'],
      ...batchResults.predictions.map((pred: any) => [
        pred.id,
        pred.prediction,
        pred.probabilities ? Math.max(...Object.values(pred.probabilities).map(Number)) || 'N/A' : 'N/A'
      ])
    ].map(row => row.join(',')).join('\n')

    const blob = new Blob([csv], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `predictions_model_${modelId}.csv`
    a.click()
    URL.revokeObjectURL(url)
  }

  if (!modelId) {
    return (
      <div className="text-center py-12">
        <BarChart3 className="w-12 h-12 mx-auto mb-4 text-gray-400" />
        <p className="text-gray-600">Select a model to start making predictions</p>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Mode Selection */}
      <div className="flex gap-4">
        <button
          onClick={() => setPredictionMode('single')}
          className={`px-4 py-2 rounded-lg font-medium ${
            predictionMode === 'single'
              ? 'bg-blue-600 text-white'
              : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
          }`}
        >
          Single Prediction
        </button>
        <button
          onClick={() => setPredictionMode('batch')}
          className={`px-4 py-2 rounded-lg font-medium ${
            predictionMode === 'batch'
              ? 'bg-blue-600 text-white'
              : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
          }`}
        >
          Batch Prediction
        </button>
      </div>

      {/* Single Prediction Form */}
      {predictionMode === 'single' && (
        <div className="space-y-4">
          <h3 className="text-lg font-semibold">Single Prediction</h3>
          
          <div className="grid grid-cols-2 gap-4">
            {featureFields.map((field) => (
              <div key={field.name}>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  {field.label}
                </label>
                <input
                  type={field.type}
                  value={features[field.name] || ''}
                  onChange={(e) => handleFeatureChange(field.name, e.target.value)}
                  placeholder={field.placeholder}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                />
              </div>
            ))}
          </div>

          <button
            onClick={handleSinglePrediction}
            disabled={loading || Object.keys(features).length === 0}
            className="flex items-center gap-2 px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? (
              <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
            ) : (
              <Play className="w-5 h-5" />
            )}
            Predict
          </button>
        </div>
      )}

      {/* Batch Prediction Form */}
      {predictionMode === 'batch' && (
        <div className="space-y-4">
          <h3 className="text-lg font-semibold">Batch Prediction</h3>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Upload CSV File
            </label>
            <div className="border-2 border-dashed border-gray-300 rounded-lg p-6">
              <input
                type="file"
                accept=".csv"
                onChange={(e) => setBatchFile(e.target.files?.[0] || null)}
                className="hidden"
                id="batch-file"
              />
              <label htmlFor="batch-file" className="cursor-pointer">
                <div className="text-center">
                  <Upload className="w-8 h-8 mx-auto mb-2 text-gray-400" />
                  <p className="text-sm text-gray-600">
                    {batchFile ? batchFile.name : 'Choose CSV file or drag and drop'}
                  </p>
                </div>
              </label>
            </div>
          </div>

          <button
            onClick={handleBatchPrediction}
            disabled={loading || !batchFile}
            className="flex items-center gap-2 px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? (
              <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
            ) : (
              <BarChart3 className="w-5 h-5" />
            )}
            Run Batch Prediction
          </button>
        </div>
      )}

      {/* Error Display */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <p className="text-red-800">{error}</p>
        </div>
      )}

      {/* Single Prediction Result */}
      {result && (
        <div className="bg-green-50 border border-green-200 rounded-lg p-6">
          <h4 className="font-semibold text-green-800 mb-4">Prediction Result</h4>
          
          <div className="space-y-3">
            <div>
              <span className="font-medium">Prediction: </span>
              <span className="text-lg font-bold text-green-700">{result.prediction}</span>
            </div>
            
            {Object.keys(result.probabilities).length > 0 && (
              <div>
                <span className="font-medium">Confidence Scores:</span>
                <div className="mt-2 space-y-1">
                  {Object.entries(result.probabilities).map(([key, value]) => (
                    <div key={key} className="flex justify-between text-sm">
                      <span>{key.replace('prob_class_', 'Class ')}</span>
                      <span className="font-mono">{(value * 100).toFixed(1)}%</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
            
            <div className="text-sm text-gray-600">
              Algorithm: {result.algorithm} | Time: {new Date(result.prediction_timestamp).toLocaleTimeString()}
            </div>
          </div>
        </div>
      )}

      {/* Batch Prediction Results */}
      {batchResults && (
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-6">
          <div className="flex items-center justify-between mb-4">
            <h4 className="font-semibold text-blue-800">
              Batch Results ({batchResults.batch_size} predictions)
            </h4>
            <button
              onClick={downloadBatchResults}
              className="flex items-center gap-2 px-3 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 text-sm"
            >
              <Download className="w-4 h-4" />
              Download CSV
            </button>
          </div>
          
          <div className="max-h-60 overflow-y-auto">
            <table className="w-full text-sm">
              <thead className="bg-blue-100">
                <tr>
                  <th className="px-3 py-2 text-left">ID</th>
                  <th className="px-3 py-2 text-left">Prediction</th>
                  <th className="px-3 py-2 text-left">Confidence</th>
                </tr>
              </thead>
              <tbody>
                {batchResults.predictions.slice(0, 10).map((pred: any) => (
                  <tr key={pred.id} className="border-t border-blue-200">
                    <td className="px-3 py-2">{pred.id}</td>
                    <td className="px-3 py-2 font-medium">{pred.prediction}</td>
                    <td className="px-3 py-2">
                      {pred.probabilities ? 
                        `${(Math.max(...Object.values(pred.probabilities).map(Number)) * 100).toFixed(1)}%` : 
                        'N/A'
                      }
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            {batchResults.predictions.length > 10 && (
              <p className="text-center text-gray-500 mt-2">
                Showing first 10 results. Download CSV for complete results.
              </p>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
