'use client'

import { useState } from 'react'
import { Upload, Play, Download, Settings, BarChart3, Database, Code, Globe, Target, FileText } from 'lucide-react'
import Image from 'next/image'

export default function Dashboard() {
  const [activeTab, setActiveTab] = useState('upload')
  const [isTraining, setIsTraining] = useState(false)
  const [logs, setLogs] = useState<string[]>([])
  const [uploadedFile, setUploadedFile] = useState<File | null>(null)
  const [showPreview, setShowPreview] = useState(false)
  const [previewData, setPreviewData] = useState<any[]>([])
  const [selectedModel, setSelectedModel] = useState('')
  const [autoMode, setAutoMode] = useState(false)
  const [modelConfig, setModelConfig] = useState({
    n_estimators: '',
    max_depth: '',
    criterion: 'gini'
  })

  const models = [
    {
      id: 'logistic_regression',
      name: 'Logistic Regression',
      description: 'A linear model for binary classification.',
      image: 'https://lh3.googleusercontent.com/aida-public/AB6AXuCJbbw6xr3tTO7843lVsaL-ryYCcUzWdMjLBowHMwzfe6vCk-FqaZPyPxQ2BOxCEUckzZQ1SkSooFT1T13D9rSA_Ie9p950kT1zcY3MiSrtm2tBZv27te29qKPUq8KVKwLRJdoIDkuUzkI1sWKbRLzEP2f7jOdhSFHZ6-aPfp42_MC_qSC7PXbNf8kd8SBnkz7U6YMPkX1LS0iOguFKDS3Ti55BwdRbWRMedgZflDgzxPNtCSKwDlwNDRE83cWogxIoNbAMWDPqp3c'
    },
    {
      id: 'random_forest',
      name: 'Random Forest',
      description: 'An ensemble of decision trees for classification and regression.',
      image: 'https://lh3.googleusercontent.com/aida-public/AB6AXuDSwdzR4PH6hmYErHh5YAQis-yPs3rYLz_ChqOsaDvEqvXUhrXNOuru6Dj-EmTfR_-e8pYK62G0vBadjLM6ASVfZ8qDz8t38iNnAimyYDPl75lUdRBsXFpQ03_iPknmI34oCDGEgPCSSkMi2_CZPVCnDoDTvE4tHD8x9pB5l8kGvBIEYz1oNhM33zuUf6vuu1Gk3O9YyGPrPrYHzAJSlfKNHAQrpzwEUDcPy8rzj8N--atSEtMvvBLcIID-5fVUqlN9zxE5dAn-iGk'
    },
    {
      id: 'svm',
      name: 'Support Vector Machine',
      description: 'A powerful model for classification and regression.',
      image: 'https://lh3.googleusercontent.com/aida-public/AB6AXuA5OkzBuRr5sABjvDhU7amWguOC-tl0RrjGPXOY2wzTajRSGQJny8iyuEqLj4tQ19meuTCCM-7dQjoQKJZ5Ug5s0blVebv5CvRhM_SsGwSCzHxY2b8c2MNBE1-ejSezKpn_rSWoy_qYvqZUykCzP21tDJgNO8ybf3qrBYk7TOXjAZqxUwWbSetX4mUuiF7Tx-7rlDLIBH40-qUuECtRD5dj6KhOXrzCfubCIGPOANU-qQTMny9HaUDJAEo3P5K46hloOgybIySZdBA'
    },
    {
      id: 'knn',
      name: 'K-Nearest Neighbors',
      description: 'A simple, non-parametric model for classification and regression.',
      image: 'https://lh3.googleusercontent.com/aida-public/AB6AXuBMdKc85t7mUXtC9QaQHRxcuYl3qC9l5MAapFmdlKXS8xQETSIvNWrCY7QL8Lu1VmqoP3QWPhCIRQVqklbaq5TkkGtUH43_lv8-eoJqMkJZUBbTtyiDrGFTv0RSaDkOJ7EtZN0rVUbE7wy-WeXo1xDP5qbjYbl-Z7kiXrEwuG3vk14JufdBXfU2Z9XtbYvXPEvL-onBa7URUv2UXskE0wO0S_ZaxRAY1u_Y_dC2FmMnS5sc7AmSeFjjQm2a25m7I1Fmsa1eSpCFRyU'
    }
  ]

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file && file.type === 'text/csv') {
      setUploadedFile(file)
      // Simulate CSV parsing and show preview
      const mockData = Array.from({ length: 10 }, (_, i) => ({
        field1: `Value ${i * 5 + 1}`,
        field2: `Value ${i * 5 + 2}`,
        field3: `Value ${i * 5 + 3}`,
        field4: `Value ${i * 5 + 4}`,
        field5: `Value ${i * 5 + 5}`,
      }))
      setPreviewData(mockData)
      setShowPreview(true)
    }
  }

  const handlePreviewConfirm = () => {
    setShowPreview(false)
    setActiveTab('train')
    console.log('Data confirmed, proceeding to training...')
  }

  const handleReupload = () => {
    setUploadedFile(null)
    setShowPreview(false)
    setPreviewData([])
  }

  const handleTrain = () => {
    setIsTraining(true)
    setLogs(['Starting model training...', 'Loading dataset...', 'Preprocessing data...'])
    
    // Simulate training logs
    setTimeout(() => {
      setLogs(prev => [...prev, 'Training model...', 'Evaluating performance...'])
    }, 2000)
    
    setTimeout(() => {
      setLogs(prev => [...prev, 'Training complete!', 'Model accuracy: 94.2%'])
      setIsTraining(false)
    }, 4000)
  }

  const navItems = [
    { id: 'upload', label: 'Upload Data', icon: Upload, color: 'blue' },
    { id: 'train', label: 'Train Model', icon: Play, color: 'green' },
    { id: 'evaluate', label: 'Evaluate', icon: Target, color: 'purple' },
    { id: 'predict', label: 'Predict', icon: BarChart3, color: 'orange' },
    { id: 'export', label: 'Export', icon: FileText, color: 'cyan' },
  ]

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-blue-900 to-green-900 flex">
      {/* Left Sidebar Navigation */}
      <div className="w-64 bg-gray-800/80 backdrop-blur-sm border-r border-gray-700 flex flex-col">
        {/* Logo Section */}
        <div className="p-6 border-b border-gray-700">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 relative">
              <Image
                src="/Screenshot 2025-07-19 at 10.33.37 AM.png"
                alt="VibeML Logo"
                width={40}
                height={40}
                className="rounded-lg object-contain"
              />
            </div>
            <div>
              <h1 className="text-xl font-bold text-white">VibeML</h1>
              <span className="text-xs text-gray-300">No-Code AutoML</span>
            </div>
          </div>
        </div>

        {/* Navigation Items */}
        <nav className="flex-1 p-4">
          <div className="space-y-2">
            {navItems.map((item) => {
              const Icon = item.icon
              const isActive = activeTab === item.id
              const colorClasses = {
                blue: isActive ? 'bg-blue-600/20 text-blue-300 border-blue-500/30' : 'text-gray-300 hover:text-blue-300 hover:bg-blue-600/10',
                green: isActive ? 'bg-green-600/20 text-green-300 border-green-500/30' : 'text-gray-300 hover:text-green-300 hover:bg-green-600/10',
                purple: isActive ? 'bg-purple-600/20 text-purple-300 border-purple-500/30' : 'text-gray-300 hover:text-purple-300 hover:bg-purple-600/10',
                orange: isActive ? 'bg-orange-600/20 text-orange-300 border-orange-500/30' : 'text-gray-300 hover:text-orange-300 hover:bg-orange-600/10',
                cyan: isActive ? 'bg-cyan-600/20 text-cyan-300 border-cyan-500/30' : 'text-gray-300 hover:text-cyan-300 hover:bg-cyan-600/10',
              }
              
              return (
                <button
                  key={item.id}
                  onClick={() => setActiveTab(item.id)}
                  className={`w-full flex items-center space-x-3 px-4 py-3 rounded-lg transition-all duration-200 ${
                    isActive ? `border ${colorClasses[item.color]}` : colorClasses[item.color]
                  }`}
                >
                  <Icon className="w-5 h-5" />
                  <span className="font-medium">{item.label}</span>
                </button>
              )
            })}
          </div>
        </nav>

        {/* Quick Stats in Sidebar */}
        <div className="p-4 border-t border-gray-700">
          <h3 className="text-sm font-semibold mb-3 text-gray-300">Quick Stats</h3>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-gray-400">Models</span>
              <span className="text-blue-400 font-medium">0</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Datasets</span>
              <span className="text-green-400 font-medium">0</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">APIs</span>
              <span className="text-cyan-400 font-medium">0</span>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col">
        {/* Main Panel */}
        <main className="flex-1 p-8 overflow-y-auto">
          <div className="max-w-6xl mx-auto">
            {activeTab === 'upload' && (
              <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg shadow-xl p-6 border border-gray-700">
                <h2 className="text-2xl font-semibold mb-6 flex items-center text-white">
                  <Database className="w-6 h-6 mr-3 text-blue-400" />
                  {showPreview ? 'Data Preview & Validation' : 'Upload Your Dataset'}
                </h2>
                
                {!showPreview ? (
                  <>
                    <div className="border-2 border-dashed border-gray-600 rounded-lg p-12 text-center hover:border-blue-400 transition-colors cursor-pointer bg-gray-900/30">
                      <input
                        type="file"
                        accept=".csv"
                        onChange={handleFileUpload}
                        className="hidden"
                        id="csv-upload"
                      />
                      <label htmlFor="csv-upload" className="cursor-pointer">
                        <Upload className="w-16 h-16 text-gray-400 mx-auto mb-4" />
                        <h3 className="text-xl font-medium text-white mb-2">
                          Drop your CSV file here
                        </h3>
                        <p className="text-gray-400 mb-6">
                          or click to browse files
                        </p>
                        <span className="bg-gradient-to-r from-blue-600 to-blue-700 text-white px-6 py-3 rounded-md hover:from-blue-700 hover:to-blue-800 transition-all font-medium inline-block">
                          Choose File
                        </span>
                      </label>
                    </div>
                    <div className="mt-8 grid grid-cols-1 md:grid-cols-3 gap-6">
                      <div className="bg-gray-700/50 p-6 rounded-lg border border-gray-600">
                        <h4 className="font-medium text-white text-lg mb-2">Supported Formats</h4>
                        <p className="text-sm text-gray-300">CSV files up to 100MB</p>
                      </div>
                      <div className="bg-gray-700/50 p-6 rounded-lg border border-gray-600">
                        <h4 className="font-medium text-white text-lg mb-2">Auto-Detection</h4>
                        <p className="text-sm text-gray-300">Automatically detects data types</p>
                      </div>
                      <div className="bg-gray-700/50 p-6 rounded-lg border border-gray-600">
                        <h4 className="font-medium text-white text-lg mb-2">Preprocessing</h4>
                        <p className="text-sm text-gray-300">Handles missing values</p>
                      </div>
                    </div>
                  </>
                ) : (
                  <>
                    <p className="text-gray-300 text-sm mb-6">
                      Here's a sample of your dataset. Please verify the structure and content before proceeding.
                    </p>
                    
                    {/* Data Preview Table */}
                    <div className="overflow-hidden rounded-lg border border-gray-600 bg-gray-800/30 mb-4">
                      <div className="overflow-x-auto">
                        <table className="w-full">
                          <thead>
                            <tr className="bg-gray-700/50">
                              <th className="px-4 py-3 text-left text-white text-sm font-medium">Field 1</th>
                              <th className="px-4 py-3 text-left text-white text-sm font-medium">Field 2</th>
                              <th className="px-4 py-3 text-left text-white text-sm font-medium">Field 3</th>
                              <th className="px-4 py-3 text-left text-white text-sm font-medium">Field 4</th>
                              <th className="px-4 py-3 text-left text-white text-sm font-medium">Field 5</th>
                            </tr>
                          </thead>
                          <tbody>
                            {previewData.map((row, index) => (
                              <tr key={index} className="border-t border-gray-600">
                                <td className="px-4 py-3 text-white text-sm">{row.field1}</td>
                                <td className="px-4 py-3 text-gray-300 text-sm">{row.field2}</td>
                                <td className="px-4 py-3 text-gray-300 text-sm">{row.field3}</td>
                                <td className="px-4 py-3 text-gray-300 text-sm">{row.field4}</td>
                                <td className="px-4 py-3 text-gray-300 text-sm">{row.field5}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                    
                    <p className="text-blue-400 text-sm mb-6 px-4">
                      10 of {previewData.length} rows shown
                    </p>
                    
                    {/* Action Buttons */}
                    <div className="flex justify-end gap-4">
                      <button
                        onClick={handlePreviewConfirm}
                        className="bg-gradient-to-r from-blue-600 to-blue-700 text-white px-6 py-3 rounded-lg hover:from-blue-700 hover:to-blue-800 transition-all font-medium"
                      >
                        Looks Good
                      </button>
                      <button
                        onClick={handleReupload}
                        className="bg-gray-600 text-white px-6 py-3 rounded-lg hover:bg-gray-700 transition-all font-medium"
                      >
                        Re-upload CSV
                      </button>
                    </div>
                  </>
                )}
              </div>
            )}

            {activeTab === 'train' && (
              <div className="flex gap-6">
                {/* Main Model Selection Panel */}
                <div className="flex-1">
                  <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg shadow-xl p-6 border border-gray-700 mb-6">
                    <div className="flex justify-between items-start mb-6">
                      <div>
                        <h2 className="text-2xl font-semibold flex items-center text-white mb-2">
                          <Play className="w-6 h-6 mr-3 text-green-400" />
                          Model Selection & Configuration
                        </h2>
                        <p className="text-gray-300 text-sm">
                          Choose a machine learning model for training. If you're unsure, we'll recommend one based on your dataset.
                        </p>
                      </div>
                    </div>
                    
                    <div className="flex justify-between mb-6">
                      <button
                        onClick={() => setActiveTab('upload')}
                        className="bg-gray-600 text-white px-4 py-2 rounded-lg hover:bg-gray-700 transition-all font-medium"
                      >
                        ← Back to Data Preview
                      </button>
                      <button
                        onClick={handleTrain}
                        disabled={!selectedModel && !autoMode}
                        className={`px-6 py-2 rounded-lg font-medium transition-all ${
                          (!selectedModel && !autoMode)
                            ? 'bg-gray-600 text-gray-400 cursor-not-allowed'
                            : 'bg-gradient-to-r from-green-600 to-green-700 text-white hover:from-green-700 hover:to-green-800'
                        }`}
                      >
                        Train Model →
                      </button>
                    </div>

                    {/* Auto Mode Toggle */}
                    <div className="bg-gray-700/30 rounded-lg p-4 mb-6 border border-gray-600">
                      <div className="flex items-center justify-between">
                        <div>
                          <h3 className="text-white font-medium mb-1">Enable Auto Mode</h3>
                          <p className="text-gray-300 text-sm">
                            Let the system automatically select the best model and parameters for your dataset.
                          </p>
                        </div>
                        <label className="relative flex h-8 w-14 cursor-pointer items-center rounded-full bg-gray-600 p-1 transition-colors has-[:checked]:bg-green-600">
                          <div className={`h-6 w-6 rounded-full bg-white transition-transform ${autoMode ? 'translate-x-6' : ''}`}></div>
                          <input
                            type="checkbox"
                            className="sr-only"
                            checked={autoMode}
                            onChange={(e) => setAutoMode(e.target.checked)}
                          />
                        </label>
                      </div>
                      {autoMode && (
                        <p className="text-green-300 text-sm mt-2">
                          Auto mode enabled. Manual configuration is disabled.
                        </p>
                      )}
                    </div>

                    {/* Model Selection Grid */}
                    {!autoMode && (
                      <>
                        <h3 className="text-xl font-semibold text-white mb-4">Select a Model</h3>
                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
                          {models.map((model) => (
                            <div
                              key={model.id}
                              onClick={() => setSelectedModel(model.id)}
                              className={`cursor-pointer rounded-lg border-2 transition-all p-4 ${
                                selectedModel === model.id
                                  ? 'border-green-500 bg-green-900/20'
                                  : 'border-gray-600 bg-gray-700/30 hover:border-gray-500'
                              }`}
                            >
                              <div
                                className="w-full aspect-square bg-center bg-no-repeat bg-cover rounded-lg mb-3"
                                style={{ backgroundImage: `url("${model.image}")` }}
                              ></div>
                              <h4 className="text-white font-medium mb-1">{model.name}</h4>
                              <p className="text-gray-300 text-sm">{model.description}</p>
                            </div>
                          ))}
                        </div>
                      </>
                    )}
                  </div>
                </div>

                {/* Model Configuration Panel */}
                {!autoMode && (
                  <div className="w-80">
                    <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg shadow-xl p-6 border border-gray-700">
                      <h3 className="text-xl font-semibold text-white mb-6">Model Configuration</h3>
                      
                      <div className="space-y-4">
                        <div>
                          <label className="block text-sm font-medium text-gray-300 mb-2">
                            Model Type
                          </label>
                          <select
                            value={selectedModel}
                            onChange={(e) => setSelectedModel(e.target.value)}
                            className="w-full p-3 border border-gray-600 rounded-lg focus:ring-2 focus:ring-green-500 focus:border-transparent bg-gray-700 text-white"
                          >
                            <option value="">Select model...</option>
                            {models.map((model) => (
                              <option key={model.id} value={model.id}>
                                {model.name}
                              </option>
                            ))}
                          </select>
                        </div>

                        {selectedModel === 'random_forest' && (
                          <>
                            <div>
                              <label className="block text-sm font-medium text-gray-300 mb-2">
                                n_estimators
                              </label>
                              <input
                                type="number"
                                value={modelConfig.n_estimators}
                                onChange={(e) => setModelConfig({...modelConfig, n_estimators: e.target.value})}
                                placeholder="100"
                                className="w-full p-3 border border-gray-600 rounded-lg focus:ring-2 focus:ring-green-500 focus:border-transparent bg-gray-700 text-white placeholder-gray-400"
                              />
                            </div>
                            <div>
                              <label className="block text-sm font-medium text-gray-300 mb-2">
                                max_depth
                              </label>
                              <input
                                type="number"
                                value={modelConfig.max_depth}
                                onChange={(e) => setModelConfig({...modelConfig, max_depth: e.target.value})}
                                placeholder="None"
                                className="w-full p-3 border border-gray-600 rounded-lg focus:ring-2 focus:ring-green-500 focus:border-transparent bg-gray-700 text-white placeholder-gray-400"
                              />
                            </div>
                            <div>
                              <label className="block text-sm font-medium text-gray-300 mb-2">
                                criterion
                              </label>
                              <select
                                value={modelConfig.criterion}
                                onChange={(e) => setModelConfig({...modelConfig, criterion: e.target.value})}
                                className="w-full p-3 border border-gray-600 rounded-lg focus:ring-2 focus:ring-green-500 focus:border-transparent bg-gray-700 text-white"
                              >
                                <option value="gini">gini</option>
                                <option value="entropy">entropy</option>
                              </select>
                            </div>
                          </>
                        )}
                      </div>

                      <div className="flex justify-end mt-6">
                        <button
                          onClick={() => setModelConfig({ n_estimators: '', max_depth: '', criterion: 'gini' })}
                          className="bg-gray-600 text-white px-4 py-2 rounded-lg hover:bg-gray-700 transition-all font-medium"
                        >
                          Reset to Default
                        </button>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            )}

            {activeTab === 'evaluate' && (
              <div className="space-y-6">
                {/* Header Section */}
                <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg shadow-xl p-6 border border-gray-700">
                  <div className="flex justify-between items-start mb-6">
                    <div>
                      <h2 className="text-2xl font-semibold flex items-center text-white mb-2">
                        <Target className="w-6 h-6 mr-3 text-purple-400" />
                        Model Analytics Dashboard
                      </h2>
                      <p className="text-gray-300 text-sm">
                        Explore detailed insights into your trained model's performance and dataset characteristics.
                      </p>
                    </div>
                  </div>
                </div>

                {/* Performance Metrics */}
                <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg shadow-xl p-6 border border-gray-700">
                  <h3 className="text-xl font-semibold text-white mb-6">Performance Metrics</h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-4">
                    <div className="bg-gray-700/50 p-6 rounded-lg border border-gray-600">
                      <h4 className="text-base font-medium text-white mb-2">Accuracy</h4>
                      <div className="text-2xl font-bold text-white mb-1">92%</div>
                      <p className="text-green-400 text-base font-medium">+2%</p>
                    </div>
                    <div className="bg-gray-700/50 p-6 rounded-lg border border-gray-600">
                      <h4 className="text-base font-medium text-white mb-2">Precision</h4>
                      <div className="text-2xl font-bold text-white mb-1">88%</div>
                      <p className="text-red-400 text-base font-medium">-1%</p>
                    </div>
                    <div className="bg-gray-700/50 p-6 rounded-lg border border-gray-600">
                      <h4 className="text-base font-medium text-white mb-2">Recall</h4>
                      <div className="text-2xl font-bold text-white mb-1">90%</div>
                      <p className="text-green-400 text-base font-medium">+3%</p>
                    </div>
                    <div className="bg-gray-700/50 p-6 rounded-lg border border-gray-600">
                      <h4 className="text-base font-medium text-white mb-2">F1 Score</h4>
                      <div className="text-2xl font-bold text-white mb-1">89%</div>
                      <p className="text-green-400 text-base font-medium">+1%</p>
                    </div>
                  </div>
                  
                  {/* ROC-AUC Score */}
                  <div className="grid grid-cols-1 lg:grid-cols-5 gap-4">
                    <div className="bg-gray-700/50 p-6 rounded-lg border border-gray-600">
                      <h4 className="text-base font-medium text-white mb-2">ROC-AUC Score</h4>
                      <div className="text-2xl font-bold text-white">0.95</div>
                    </div>
                  </div>
                </div>

                {/* Model Overview */}
                <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg shadow-xl p-6 border border-gray-700">
                  <h3 className="text-xl font-semibold text-white mb-6">Model Overview</h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-x-4">
                    <div className="flex flex-col gap-1 border-t border-gray-600 py-4">
                      <p className="text-gray-400 text-sm">Model Type</p>
                      <p className="text-white text-sm">Classification</p>
                    </div>
                    <div className="flex flex-col gap-1 border-t border-gray-600 py-4">
                      <p className="text-gray-400 text-sm">Training Time</p>
                      <p className="text-white text-sm">2 hours 30 minutes</p>
                    </div>
                    <div className="flex flex-col gap-1 border-t border-gray-600 py-4">
                      <p className="text-gray-400 text-sm">Number of Iterations/Epochs</p>
                      <p className="text-white text-sm">100 epochs</p>
                    </div>
                    <div className="flex flex-col gap-1 border-t border-gray-600 py-4">
                      <p className="text-gray-400 text-sm">Algorithm Used</p>
                      <p className="text-white text-sm">Gradient Boosting</p>
                    </div>
                    <div className="flex flex-col gap-1 border-t border-gray-600 py-4">
                      <p className="text-gray-400 text-sm">Dataset Name</p>
                      <p className="text-white text-sm">Customer Retention Dataset</p>
                    </div>
                    <div className="flex flex-col gap-1 border-t border-gray-600 py-4">
                      <p className="text-gray-400 text-sm">Dataset Size</p>
                      <p className="text-white text-sm">10,000 rows</p>
                    </div>
                  </div>
                </div>

                {/* Interactive Charts */}
                <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg shadow-xl border border-gray-700">
                  <div className="p-6 pb-0">
                    <h3 className="text-xl font-semibold text-white mb-6">Interactive Charts</h3>
                  </div>
                  
                  {/* Chart Tabs */}
                  <div className="border-b border-gray-600">
                    <div className="flex px-6 gap-8">
                      <button className="flex flex-col items-center justify-center border-b-2 border-blue-400 text-white pb-3 pt-4">
                        <span className="text-white text-sm font-bold">ROC Curve</span>
                      </button>
                      <button className="flex flex-col items-center justify-center border-b-2 border-transparent text-gray-400 pb-3 pt-4 hover:text-white">
                        <span className="text-gray-400 text-sm font-bold">Precision-Recall</span>
                      </button>
                      <button className="flex flex-col items-center justify-center border-b-2 border-transparent text-gray-400 pb-3 pt-4 hover:text-white">
                        <span className="text-gray-400 text-sm font-bold">Feature Importance</span>
                      </button>
                      <button className="flex flex-col items-center justify-center border-b-2 border-transparent text-gray-400 pb-3 pt-4 hover:text-white">
                        <span className="text-gray-400 text-sm font-bold">Loss vs Epoch</span>
                      </button>
                    </div>
                  </div>
                  
                  {/* Chart Content */}
                  <div className="p-6">
                    <div className="bg-gray-700/30 rounded-lg p-6 border border-gray-600">
                      <div className="flex justify-between items-start mb-4">
                        <div>
                          <h4 className="text-white font-medium mb-1">ROC Curve</h4>
                          <div className="text-2xl font-bold text-white mb-2">0.95</div>
                          <div className="flex gap-2 items-center">
                            <span className="text-gray-300 text-sm">Overall</span>
                            <span className="text-green-400 text-sm font-medium">+2%</span>
                          </div>
                        </div>
                      </div>
                      
                      {/* Placeholder Chart */}
                      <div className="h-48 bg-gray-800/50 rounded-lg border border-gray-600 flex items-center justify-center relative overflow-hidden">
                        {/* Simple SVG curve simulation */}
                        <svg className="absolute inset-0 w-full h-full" viewBox="0 0 400 200">
                          <defs>
                            <linearGradient id="rocGradient" x1="0%" y1="0%" x2="0%" y2="100%">
                              <stop offset="0%" stopColor="rgb(59, 130, 246)" stopOpacity="0.3"/>
                              <stop offset="100%" stopColor="rgb(59, 130, 246)" stopOpacity="0.1"/>
                            </linearGradient>
                          </defs>
                          <path
                            d="M 20 180 Q 100 160 200 100 T 380 20"
                            stroke="rgb(59, 130, 246)"
                            strokeWidth="2"
                            fill="none"
                          />
                          <path
                            d="M 20 180 Q 100 160 200 100 T 380 20 L 380 180 L 20 180 Z"
                            fill="url(#rocGradient)"
                          />
                        </svg>
                        <div className="absolute bottom-2 left-0 right-0 flex justify-between px-4 text-xs text-gray-400">
                          <span>0.0</span>
                          <span>0.2</span>
                          <span>0.4</span>
                          <span>0.6</span>
                          <span>0.8</span>
                          <span>1.0</span>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Dataset Summary */}
                <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg shadow-xl p-6 border border-gray-700">
                  <h3 className="text-xl font-semibold text-white mb-4">Dataset Summary</h3>
                  <p className="text-white text-base mb-6">Key insights about the dataset used for training:</p>
                  
                  <div className="space-y-4">
                    <div className="bg-gray-700/30 p-4 rounded-lg border border-gray-600">
                      <h4 className="text-white font-medium mb-1">Null Values</h4>
                      <p className="text-gray-300 text-sm">
                        15% of the dataset contains missing values, primarily in the 'Age' and 'Income' columns.
                      </p>
                    </div>
                    <div className="bg-gray-700/30 p-4 rounded-lg border border-gray-600">
                      <h4 className="text-white font-medium mb-1">Dominant Class</h4>
                      <p className="text-gray-300 text-sm">
                        The 'Retention' class is the dominant class, representing 70% of the dataset.
                      </p>
                    </div>
                    <div className="bg-gray-700/30 p-4 rounded-lg border border-gray-600">
                      <h4 className="text-white font-medium mb-1">Class Imbalance</h4>
                      <p className="text-gray-300 text-sm">
                        The dataset exhibits a moderate class imbalance, with a ratio of approximately 2:1 between the 'Retention' and 'No Retention' classes.
                      </p>
                    </div>
                  </div>

                  {/* Distribution Charts */}
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-6">
                    <div className="bg-gray-700/30 p-6 rounded-lg border border-gray-600">
                      <h4 className="text-white font-medium mb-2">Distribution of Categorical Variable A</h4>
                      <div className="text-2xl font-bold text-white mb-2">60%</div>
                      <div className="flex gap-2 items-center mb-4">
                        <span className="text-gray-300 text-sm">Overall</span>
                        <span className="text-green-400 text-sm font-medium">+5%</span>
                      </div>
                      
                      {/* Bar Chart */}
                      <div className="h-32 flex items-end justify-center gap-4">
                        <div className="flex flex-col items-center">
                          <div className="w-12 bg-blue-500 border-t-2 border-blue-400" style={{height: '60%'}}></div>
                          <span className="text-xs text-gray-400 mt-2">Category 1</span>
                        </div>
                        <div className="flex flex-col items-center">
                          <div className="w-12 bg-blue-500 border-t-2 border-blue-400" style={{height: '40%'}}></div>
                          <span className="text-xs text-gray-400 mt-2">Category 2</span>
                        </div>
                      </div>
                    </div>
                    
                    <div className="bg-gray-700/30 p-6 rounded-lg border border-gray-600">
                      <h4 className="text-white font-medium mb-2">Distribution of Categorical Variable B</h4>
                      <div className="text-2xl font-bold text-white mb-2">40%</div>
                      <div className="flex gap-2 items-center mb-4">
                        <span className="text-gray-300 text-sm">Overall</span>
                        <span className="text-red-400 text-sm font-medium">-5%</span>
                      </div>
                      
                      {/* Bar Chart */}
                      <div className="h-32 flex items-end justify-center gap-4">
                        <div className="flex flex-col items-center">
                          <div className="w-8 bg-purple-500 border-t-2 border-purple-400" style={{height: '80%'}}></div>
                          <span className="text-xs text-gray-400 mt-2">A</span>
                        </div>
                        <div className="flex flex-col items-center">
                          <div className="w-8 bg-purple-500 border-t-2 border-purple-400" style={{height: '50%'}}></div>
                          <span className="text-xs text-gray-400 mt-2">B</span>
                        </div>
                        <div className="flex flex-col items-center">
                          <div className="w-8 bg-purple-500 border-t-2 border-purple-400" style={{height: '100%'}}></div>
                          <span className="text-xs text-gray-400 mt-2">C</span>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Model Retraining */}
                <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg shadow-xl p-6 border border-gray-700">
                  <h3 className="text-xl font-semibold text-white mb-4">Model Retraining (Optional)</h3>
                  <p className="text-white text-base mb-6">
                    Recommended improvements: Optimize hyperparameters for better performance.
                  </p>
                  
                  <div className="mb-6">
                    <button className="bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700 transition-all font-medium">
                      Retrain with optimized hyperparameters
                    </button>
                  </div>

                  {/* Configuration Table */}
                  <div className="overflow-hidden rounded-lg border border-gray-600">
                    <table className="w-full">
                      <thead>
                        <tr className="bg-gray-700/50">
                          <th className="px-4 py-3 text-left text-white text-sm font-medium">Configuration</th>
                          <th className="px-4 py-3 text-left text-white text-sm font-medium">Hyperparameter A</th>
                          <th className="px-4 py-3 text-left text-white text-sm font-medium">Hyperparameter B</th>
                          <th className="px-4 py-3 text-left text-white text-sm font-medium">Result</th>
                        </tr>
                      </thead>
                      <tbody>
                        <tr className="border-t border-gray-600">
                          <td className="px-4 py-3 text-white text-sm">Config 1</td>
                          <td className="px-4 py-3 text-gray-300 text-sm">Value 1</td>
                          <td className="px-4 py-3 text-gray-300 text-sm">Value 2</td>
                          <td className="px-4 py-3 text-gray-300 text-sm">90%</td>
                        </tr>
                        <tr className="border-t border-gray-600">
                          <td className="px-4 py-3 text-white text-sm">Config 2</td>
                          <td className="px-4 py-3 text-gray-300 text-sm">Value 3</td>
                          <td className="px-4 py-3 text-gray-300 text-sm">Value 4</td>
                          <td className="px-4 py-3 text-gray-300 text-sm">91%</td>
                        </tr>
                        <tr className="border-t border-gray-600">
                          <td className="px-4 py-3 text-white text-sm">Config 3</td>
                          <td className="px-4 py-3 text-gray-300 text-sm">Value 5</td>
                          <td className="px-4 py-3 text-gray-300 text-sm">Value 6</td>
                          <td className="px-4 py-3 text-gray-300 text-sm">92%</td>
                        </tr>
                      </tbody>
                    </table>
                  </div>
                </div>

                {/* Export & Share */}
                <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg shadow-xl p-6 border border-gray-700">
                  <h3 className="text-xl font-semibold text-white mb-6">Export & Share</h3>
                  <div className="flex gap-4">
                    <button className="bg-gray-600 text-white px-6 py-2 rounded-lg hover:bg-gray-700 transition-all font-medium">
                      Download PDF Report
                    </button>
                    <button className="bg-gray-600 text-white px-6 py-2 rounded-lg hover:bg-gray-700 transition-all font-medium">
                      Share Dashboard
                    </button>
                  </div>
                </div>
              </div>
            )}

            {activeTab === 'predict' && (
              <div className="flex flex-row gap-8">
                {/* Main Prediction Section */}
                <div className="flex flex-col flex-1 max-w-3xl">
                  <div className="flex flex-wrap justify-between gap-3 mb-6">
                    <div className="flex min-w-72 flex-col gap-3">
                      <p className="text-white text-3xl font-bold leading-tight">📊 Predictions</p>
                      <p className="text-blue-300 text-sm font-normal">Use your trained model to make predictions on new data.</p>
                    </div>
                    <button className="bg-gray-700 text-white px-6 py-2 rounded-lg font-medium hover:bg-gray-600 transition-all">
                      Upload New Dataset for Prediction
                    </button>
                  </div>
                  <div className="mb-6">
                    <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4 rounded-lg border border-gray-700 bg-gray-900 p-5">
                      <div className="flex flex-col gap-1">
                        <p className="text-white text-base font-bold">Prediction Options</p>
                        <p className="text-blue-300 text-base font-normal">Step 1: Upload a CSV file to make predictions</p>
                      </div>
                      <button className="bg-blue-600 text-white px-6 py-2 rounded-lg font-medium hover:bg-blue-700 transition-all">
                        Upload CSV
                      </button>
                    </div>
                  </div>
                  <div className="mb-6">
                    <div className="flex items-stretch justify-between gap-4 rounded-lg">
                      <div className="flex flex-col gap-1 flex-[2_2_0px]">
                        <p className="text-white text-base font-bold">Model Used: Random Forest Classifier</p>
                        <p className="text-blue-300 text-sm font-normal">Last Trained: 2024-01-15 | Training Accuracy: 92% | Dataset: Customer Churn Data</p>
                      </div>
                      <div className="hidden md:block w-full max-w-xs aspect-video bg-gray-700 rounded-lg border border-gray-600"></div>
                    </div>
                  </div>
                  {/* Prediction Table */}
                  <div className="mb-6">
                    <div className="overflow-x-auto rounded-lg border border-gray-700 bg-gray-900">
                      <table className="min-w-full">
                        <thead>
                          <tr className="bg-gray-900">
                            <th className="px-4 py-3 text-left text-white text-sm font-medium">Row ID</th>
                            <th className="px-4 py-3 text-left text-white text-sm font-medium">Original Inputs</th>
                            <th className="px-4 py-3 text-left text-white text-sm font-medium">Prediction</th>
                            <th className="px-4 py-3 text-left text-white text-sm font-medium">Confidence</th>
                            <th className="px-4 py-3 text-left text-white text-sm font-medium">Probability</th>
                          </tr>
                        </thead>
                        <tbody>
                          {[1,2,3,4,5,6,7,8,9,10].map((row) => (
                            <tr key={row} className="border-t border-gray-700">
                              <td className="px-4 py-2 text-white text-sm">{row}</td>
                              <td className="px-4 py-2 text-blue-300 text-sm">{'{feature1: value1, feature2: value2, ...}'}</td>
                              <td className="px-4 py-2 text-sm">
                                <button className="bg-gray-700 text-white px-4 py-2 rounded-lg font-medium w-full">
                                  <span className="truncate">{row % 2 === 0 ? 'No Churn' : 'Churn'}</span>
                                </button>
                              </td>
                              <td className="px-4 py-2 text-sm">
                                <div className="flex items-center gap-3">
                                  <div className="w-20 h-1 rounded-full bg-gray-600 overflow-hidden">
                                    <div className="h-1 rounded-full bg-blue-600" style={{width: `${80 + row}%`}}></div>
                                  </div>
                                  <p className="text-white text-sm font-medium">{80 + row}</p>
                                </div>
                              </td>
                              <td className="px-4 py-2 text-sm">
                                <div className="flex items-center gap-3">
                                  <div className="w-20 h-1 rounded-full bg-gray-600 overflow-hidden">
                                    <div className="h-1 rounded-full bg-blue-600" style={{width: `${85 + row}%`}}></div>
                                  </div>
                                  <p className="text-white text-sm font-medium">{85 + row}</p>
                                </div>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                  {/* Pagination */}
                  <div className="flex items-center justify-center mb-6 gap-2">
                    <button className="flex items-center justify-center w-8 h-8 rounded-full bg-gray-700 text-white">
                      {'<'}
                    </button>
                    <button className="w-8 h-8 rounded-full bg-blue-600 text-white font-bold">1</button>
                    <button className="w-8 h-8 rounded-full bg-gray-700 text-white">2</button>
                    <button className="w-8 h-8 rounded-full bg-gray-700 text-white">3</button>
                    <button className="flex items-center justify-center w-8 h-8 rounded-full bg-gray-700 text-white">
                      {'>'}
                    </button>
                  </div>
                </div>
                {/* Insights & Export Section */}
                <div className="flex flex-col w-[360px]">
                  <h2 className="text-white text-xl font-bold px-4 pb-3 pt-5">Insights</h2>
                  <div className="flex flex-wrap gap-4 px-4 py-6">
                    <div className="flex min-w-72 flex-1 flex-col gap-2 rounded-lg border border-gray-700 p-6 bg-gray-900">
                      <p className="text-white text-base font-medium">Distribution of Predicted Values</p>
                      <p className="text-white text-3xl font-bold truncate">20</p>
                      <div className="flex gap-1">
                        <p className="text-blue-300 text-base font-normal">Total</p>
                        <p className="text-green-400 text-base font-medium">+5%</p>
                      </div>
                      <div className="grid min-h-[180px] grid-flow-col gap-6 grid-rows-[1fr_auto] items-end justify-items-center px-3">
                        <div className="border-blue-300 bg-gray-700 border-t-2 w-full" style={{height: '40%'}}></div>
                        <p className="text-blue-300 text-[13px] font-bold">Churn</p>
                        <div className="border-blue-300 bg-gray-700 border-t-2 w-full" style={{height: '20%'}}></div>
                        <p className="text-blue-300 text-[13px] font-bold">No Churn</p>
                      </div>
                    </div>
                    <div className="flex min-w-72 flex-1 flex-col gap-2 rounded-lg border border-gray-700 p-6 bg-gray-900">
                      <p className="text-white text-base font-medium">Confidence Score Histogram</p>
                      <p className="text-white text-3xl font-bold truncate">85%</p>
                      <div className="flex gap-1">
                        <p className="text-blue-300 text-base font-normal">Average</p>
                        <p className="text-red-400 text-base font-medium">-2%</p>
                      </div>
                      <div className="grid min-h-[180px] gap-x-4 gap-y-6 grid-cols-[auto_1fr] items-center py-3">
                        <p className="text-blue-300 text-[13px] font-bold">70-80%</p>
                        <div className="h-full flex-1"><div className="border-blue-300 bg-gray-700 border-r-2 h-full" style={{width: '40%'}}></div></div>
                        <p className="text-blue-300 text-[13px] font-bold">80-90%</p>
                        <div className="h-full flex-1"><div className="border-blue-300 bg-gray-700 border-r-2 h-full" style={{width: '60%'}}></div></div>
                        <p className="text-blue-300 text-[13px] font-bold">90-100%</p>
                        <div className="h-full flex-1"><div className="border-blue-300 bg-gray-700 border-r-2 h-full" style={{width: '10%'}}></div></div>
                      </div>
                    </div>
                  </div>
                  <h2 className="text-white text-xl font-bold px-4 pb-3 pt-5">Warnings</h2>
                  <p className="text-white text-base font-normal pb-3 pt-1 px-4">No warnings to display.</p>
                  <h2 className="text-white text-xl font-bold px-4 pb-3 pt-5">Download Options</h2>
                  <div className="flex justify-center">
                    <div className="flex flex-1 gap-3 max-w-xs flex-col items-stretch px-4 py-3">
                      <button className="bg-gray-700 text-white px-6 py-2 rounded-lg font-bold w-full">Export as CSV</button>
                      <button className="bg-gray-700 text-white px-6 py-2 rounded-lg font-bold w-full">Export as JSON</button>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {activeTab === 'export' && (
              <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg shadow-xl p-6 border border-gray-700">
                <h2 className="text-2xl font-semibold mb-6 flex items-center text-white">
                  <FileText className="w-6 h-6 mr-3 text-cyan-400" />
                  Export & Deploy
                </h2>
                <div className="space-y-6">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <button className="p-8 border-2 border-gray-600 rounded-lg hover:border-blue-500 hover:bg-blue-900/20 transition-all text-left bg-gray-900/30">
                      <Code className="w-10 h-10 text-blue-400 mb-4" />
                      <h3 className="font-medium text-white text-lg mb-2">Export Code</h3>
                      <p className="text-sm text-gray-300">
                        Download Python script with your trained model
                      </p>
                    </button>
                    <button className="p-8 border-2 border-gray-600 rounded-lg hover:border-green-500 hover:bg-green-900/20 transition-all text-left bg-gray-900/30">
                      <Globe className="w-10 h-10 text-green-400 mb-4" />
                      <h3 className="font-medium text-white text-lg mb-2">Deploy API</h3>
                      <p className="text-sm text-gray-300">
                        Create REST API endpoint for predictions
                      </p>
                    </button>
                  </div>
                  <div className="bg-gray-900/50 p-6 rounded-lg border border-gray-600">
                    <h4 className="font-medium text-white mb-3 text-lg">API Endpoint</h4>
                    <code className="text-sm bg-gray-800 text-green-300 p-3 rounded border border-gray-600 block">
                      https://api.vibeml.com/v1/predict/your-model-id
                    </code>
                  </div>
                </div>
              </div>
            )}
          </div>
        </main>

        {/* Bottom Training Logs Panel */}
        <div className="border-t border-gray-700 bg-gray-800/50 backdrop-blur-sm">
          <div className="max-w-6xl mx-auto p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-white flex items-center">
                <Settings className="w-5 h-5 mr-2 text-gray-400" />
                Training Logs
              </h3>
              <div className="flex space-x-4">
                <button className="text-gray-400 hover:text-white transition-colors">
                  <Download className="w-4 h-4" />
                </button>
                <button className="text-gray-400 hover:text-white transition-colors">
                  <Settings className="w-4 h-4" />
                </button>
              </div>
            </div>
            <div className="bg-gray-900 rounded-md p-4 h-32 overflow-y-auto border border-gray-700 font-mono text-sm">
              {logs.length === 0 ? (
                <p className="text-gray-400">No logs yet. Start training to see real-time progress...</p>
              ) : (
                <div className="space-y-1">
                  {logs.map((log, index) => (
                    <p key={index} className="text-green-400">
                      <span className="text-gray-500">[{new Date().toLocaleTimeString()}]</span> {log}
                    </p>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
