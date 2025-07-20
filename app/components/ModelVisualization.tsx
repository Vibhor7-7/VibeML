'use client'

import { useState, useEffect } from 'react'
import { BarChart3, TrendingUp, Target, Zap } from 'lucide-react'

interface ModelVisualizationProps {
  modelAnalytics: any
  modelId: string
}

export default function ModelVisualization({ modelAnalytics, modelId }: ModelVisualizationProps) {
  if (!modelAnalytics) {
    return (
      <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg shadow-xl p-6 border border-gray-700">
        <div className="text-center text-gray-400">
          <BarChart3 className="w-12 h-12 mx-auto mb-4" />
          <p>No visualization data available</p>
        </div>
      </div>
    )
  }

  // Create a simple metric bar chart
  const renderMetricBar = (label: string, value: number, color: string) => {
    const percentage = Math.round(value * 100)
    const colorClasses = {
      'blue': 'bg-blue-500',
      'green': 'bg-green-500',
      'purple': 'bg-purple-500',
      'orange': 'bg-orange-500',
      'red': 'bg-red-500'
    }

    return (
      <div key={label} className="mb-4">
        <div className="flex justify-between items-center mb-2">
          <span className="text-sm font-medium text-gray-300">{label}</span>
          <span className="text-sm text-white font-bold">{percentage}%</span>
        </div>
        <div className="w-full bg-gray-700 rounded-full h-3">
          <div 
            className={`h-3 rounded-full ${colorClasses[color]} transition-all duration-500 ease-out`}
            style={{ width: `${percentage}%` }}
          ></div>
        </div>
      </div>
    )
  }

  // Feature importance chart
  const renderFeatureImportance = () => {
    if (!modelAnalytics.feature_importance || modelAnalytics.feature_importance.length === 0) {
      return null
    }

    return (
      <div className="bg-gray-700/50 p-4 rounded-lg border border-gray-600">
        <h4 className="text-lg font-semibold text-white mb-4 flex items-center">
          <Target className="w-5 h-5 mr-2 text-purple-400" />
          Feature Importance
        </h4>
        <div className="space-y-3">
          {modelAnalytics.feature_importance.slice(0, 6).map((feature: any, index: number) => {
            const importance = feature.importance || feature.value || 0
            const percentage = Math.round(importance * 100)
            
            return (
              <div key={index} className="flex items-center space-x-3">
                <div className="w-24 text-sm text-gray-300 truncate">
                  {feature.feature || feature.name}
                </div>
                <div className="flex-1 bg-gray-600 rounded-full h-2">
                  <div 
                    className="bg-gradient-to-r from-purple-500 to-blue-500 h-2 rounded-full transition-all duration-500"
                    style={{ width: `${percentage}%` }}
                  ></div>
                </div>
                <div className="w-12 text-sm text-white font-medium">
                  {percentage}%
                </div>
              </div>
            )
          })}
        </div>
      </div>
    )
  }

  // Performance metrics visualization
  const renderMetricsChart = () => {
    const metrics = modelAnalytics.metrics || {}
    const metricsToShow = [
      { key: 'accuracy', label: 'Accuracy', color: 'green' },
      { key: 'precision', label: 'Precision', color: 'blue' },
      { key: 'recall', label: 'Recall', color: 'purple' },
      { key: 'f1_score', label: 'F1 Score', color: 'orange' }
    ].filter(m => metrics[m.key] !== undefined)

    if (metricsToShow.length === 0) {
      return null
    }

    return (
      <div className="bg-gray-700/50 p-4 rounded-lg border border-gray-600">
        <h4 className="text-lg font-semibold text-white mb-4 flex items-center">
          <TrendingUp className="w-5 h-5 mr-2 text-green-400" />
          Performance Metrics
        </h4>
        <div className="space-y-4">
          {metricsToShow.map(metric => 
            renderMetricBar(metric.label, metrics[metric.key], metric.color)
          )}
        </div>
      </div>
    )
  }

  // Training summary
  const renderTrainingSummary = () => {
    const training = modelAnalytics.training_details || {}
    
    return (
      <div className="bg-gray-700/50 p-4 rounded-lg border border-gray-600">
        <h4 className="text-lg font-semibold text-white mb-4 flex items-center">
          <Zap className="w-5 h-5 mr-2 text-yellow-400" />
          Training Summary
        </h4>
        <div className="grid grid-cols-2 gap-4 text-sm">
          <div>
            <span className="text-gray-400">Algorithm:</span>
            <span className="text-white ml-2 font-medium">
              {modelAnalytics.algorithm || 'Unknown'}
            </span>
          </div>
          <div>
            <span className="text-gray-400">Training Time:</span>
            <span className="text-white ml-2 font-medium">
              {training.training_time || 'N/A'}
            </span>
          </div>
          <div>
            <span className="text-gray-400">Dataset:</span>
            <span className="text-white ml-2 font-medium">
              {training.dataset_name || 'Unknown'}
            </span>
          </div>
          <div>
            <span className="text-gray-400">CV Score:</span>
            <span className="text-white ml-2 font-medium">
              {training.cv_score ? `${(training.cv_score * 100).toFixed(1)}%` : 'N/A'}
            </span>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {renderMetricsChart()}
        {renderFeatureImportance()}
      </div>
      {renderTrainingSummary()}
    </div>
  )
}
