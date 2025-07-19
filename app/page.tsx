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
    // Here you would typically process the data further
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
              <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg shadow-xl p-6 border border-gray-700">
                <h2 className="text-2xl font-semibold mb-6 flex items-center text-white">
                  <Play className="w-6 h-6 mr-3 text-green-400" />
                  Train Your Model
                </h2>
                <div className="space-y-6">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div>
                      <label className="block text-sm font-medium text-gray-300 mb-3">
                        Problem Type
                      </label>
                      <select className="w-full p-4 border border-gray-600 rounded-md focus:ring-2 focus:ring-green-500 focus:border-transparent bg-gray-700 text-white">
                        <option>Classification</option>
                        <option>Regression</option>
                        <option>Time Series</option>
                      </select>
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-gray-300 mb-3">
                        Target Column
                      </label>
                      <select className="w-full p-4 border border-gray-600 rounded-md focus:ring-2 focus:ring-green-500 focus:border-transparent bg-gray-700 text-white">
                        <option>Select target column...</option>
                        <option>price</option>
                        <option>category</option>
                        <option>outcome</option>
                      </select>
                    </div>
                  </div>
                  
                  <button
                    onClick={handleTrain}
                    disabled={isTraining}
                    className={`w-full py-4 px-8 rounded-md font-medium transition-all text-lg ${
                      isTraining
                        ? 'bg-gray-600 cursor-not-allowed'
                        : 'bg-gradient-to-r from-green-600 to-green-700 hover:from-green-700 hover:to-green-800 shadow-lg'
                    } text-white`}
                  >
                    {isTraining ? (
                      <span className="flex items-center justify-center">
                        <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white mr-3"></div>
                        Training Model...
                      </span>
                    ) : (
                      'Start Training'
                    )}
                  </button>
                </div>
              </div>
            )}

            {activeTab === 'evaluate' && (
              <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg shadow-xl p-6 border border-gray-700">
                <h2 className="text-2xl font-semibold mb-6 flex items-center text-white">
                  <Target className="w-6 h-6 mr-3 text-purple-400" />
                  Model Evaluation
                </h2>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                  <div className="bg-gray-700/50 p-6 rounded-lg border border-gray-600">
                    <h3 className="text-lg font-medium text-white mb-3">Accuracy</h3>
                    <div className="text-3xl font-bold text-purple-400">94.2%</div>
                    <p className="text-sm text-gray-300 mt-2">Overall model accuracy</p>
                  </div>
                  <div className="bg-gray-700/50 p-6 rounded-lg border border-gray-600">
                    <h3 className="text-lg font-medium text-white mb-3">Precision</h3>
                    <div className="text-3xl font-bold text-blue-400">91.8%</div>
                    <p className="text-sm text-gray-300 mt-2">Precision score</p>
                  </div>
                  <div className="bg-gray-700/50 p-6 rounded-lg border border-gray-600">
                    <h3 className="text-lg font-medium text-white mb-3">Recall</h3>
                    <div className="text-3xl font-bold text-green-400">89.5%</div>
                    <p className="text-sm text-gray-300 mt-2">Recall score</p>
                  </div>
                </div>
              </div>
            )}

            {activeTab === 'predict' && (
              <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg shadow-xl p-6 border border-gray-700">
                <h2 className="text-2xl font-semibold mb-6 flex items-center text-white">
                  <BarChart3 className="w-6 h-6 mr-3 text-orange-400" />
                  Make Predictions
                </h2>
                <div className="space-y-6">
                  <div className="border-2 border-dashed border-gray-600 rounded-lg p-8 text-center bg-gray-900/30">
                    <Upload className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                    <h3 className="text-lg font-medium text-white mb-2">Upload New Data</h3>
                    <p className="text-gray-400 mb-4">Upload CSV file to make predictions</p>
                    <button className="bg-gradient-to-r from-orange-600 to-orange-700 text-white px-6 py-3 rounded-md hover:from-orange-700 hover:to-orange-800 transition-all">
                      Choose File
                    </button>
                  </div>
                  <div className="bg-gray-700/50 p-6 rounded-lg border border-gray-600">
                    <h3 className="text-lg font-medium text-white mb-4">Prediction Results</h3>
                    <p className="text-gray-300">Upload data to see predictions here...</p>
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
