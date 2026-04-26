import 'dart:io';
import 'dart:ui';
import 'package:flutter/material.dart';
import '../services/api_service.dart';
import '../theme/app_theme.dart';

class AnalysisScreen extends StatefulWidget {
  final String videoPath;
  const AnalysisScreen({super.key, required this.videoPath});

  @override
  State<AnalysisScreen> createState() => _AnalysisScreenState();
}

class _AnalysisScreenState extends State<AnalysisScreen> with SingleTickerProviderStateMixin {
  final ApiService _apiService = ApiService();
  bool _isAnalyzing = true;
  double _uploadProgress = 0.0;
  Map<String, dynamic>? _result;
  String? _error;
  late AnimationController _animController;

  @override
  void initState() {
    super.initState();
    _animController = AnimationController(vsync: this, duration: const Duration(seconds: 3))..repeat();
    _startAnalysis();
  }

  void _startAnalysis() async {
    try {
      final result = await _apiService.analyzeVideo(File(widget.videoPath), (sent, total) {
        setState(() {
          _uploadProgress = sent / total;
        });
      });
      setState(() {
        _result = result;
        _isAnalyzing = false;
      });
    } catch (e) {
      setState(() {
        _error = e.toString();
        _isAnalyzing = false;
      });
    }
  }

  @override
  void dispose() {
    _animController.dispose();
    super.dispose();
  }

  Widget _buildGlassCard({required Widget child, Color? borderColor}) {
    return ClipRRect(
      borderRadius: BorderRadius.circular(16),
      child: BackdropFilter(
        filter: ImageFilter.blur(sigmaX: 10, sigmaY: 10),
        child: Container(
          padding: const EdgeInsets.all(20),
          decoration: BoxDecoration(
            color: AppTheme.surface.withOpacity(0.5),
            borderRadius: BorderRadius.circular(16),
            border: Border.all(color: borderColor ?? Colors.white.withOpacity(0.1), width: 1),
          ),
          child: child,
        ),
      ),
    );
  }

  Widget _buildMetricRow(String title, String value, Color color) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 8.0),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(title, style: const TextStyle(color: AppTheme.textMuted, fontSize: 16)),
          Text(value, style: TextStyle(color: color, fontSize: 20, fontWeight: FontWeight.bold, fontFamily: 'monospace')),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppTheme.background,
      appBar: AppBar(
        title: const Text('ANALYSIS RESULT'),
        leading: IconButton(
          icon: const Icon(Icons.arrow_back, color: AppTheme.primary),
          onPressed: () => Navigator.pop(context),
        ),
      ),
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(24.0),
          child: _isAnalyzing
              ? _buildAnalyzingState()
              : _error != null
                  ? _buildErrorState()
                  : _buildResultState(),
        ),
      ),
    );
  }

  Widget _buildAnalyzingState() {
    return Column(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        Stack(
          alignment: Alignment.center,
          children: [
            SizedBox(
              width: 150,
              height: 150,
              child: CircularProgressIndicator(
                value: _uploadProgress,
                backgroundColor: AppTheme.surface,
                color: AppTheme.primary,
                strokeWidth: 2,
              ),
            ),
            RotationTransition(
              turns: _animController,
              child: Container(
                width: 100,
                height: 100,
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  border: Border.all(color: AppTheme.accent.withOpacity(0.5), width: 2, strokeAlign: BorderSide.strokeAlignOutside),
                ),
                child: const Icon(Icons.psychology, size: 40, color: AppTheme.primary),
              ),
            ),
          ],
        ),
        const SizedBox(height: 48),
        const Text('PROCESSING NEURAL DATA...', style: TextStyle(color: AppTheme.primary, letterSpacing: 2, fontWeight: FontWeight.bold)),
        const SizedBox(height: 16),
        Text('${(_uploadProgress * 100).toStringAsFixed(1)}% UPLOADED', style: const TextStyle(color: AppTheme.textMuted, letterSpacing: 1)),
      ],
    );
  }

  Widget _buildErrorState() {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          const Icon(Icons.warning_amber_rounded, color: AppTheme.deception, size: 80),
          const SizedBox(height: 24),
          Text('ANALYSIS FAILED', style: Theme.of(context).textTheme.titleLarge?.copyWith(color: AppTheme.deception, letterSpacing: 2)),
          const SizedBox(height: 16),
          Text(_error!, textAlign: TextAlign.center, style: const TextStyle(color: AppTheme.textMuted)),
          const SizedBox(height: 32),
          ElevatedButton(
            onPressed: () => Navigator.pop(context),
            style: ElevatedButton.styleFrom(backgroundColor: AppTheme.surface, foregroundColor: Colors.white),
            child: const Text('RETURN TO SCANNER', style: TextStyle(letterSpacing: 1.5)),
          ),
        ],
      ),
    );
  }

  Widget _buildResultState() {
    final bool isDeceptive = _result!['verdict'] == 'Deceptive';
    final Color verdictColor = isDeceptive ? AppTheme.deception : AppTheme.truth;

    return ListView(
      physics: const BouncingScrollPhysics(),
      children: [
        _buildGlassCard(
          borderColor: verdictColor.withOpacity(0.5),
          child: Column(
            children: [
              const Text('FINAL VERDICT', style: TextStyle(color: AppTheme.textMuted, letterSpacing: 3, fontSize: 12)),
              const SizedBox(height: 16),
              Text(
                _result!['verdict'].toUpperCase(),
                style: Theme.of(context).textTheme.displayLarge?.copyWith(
                  color: verdictColor,
                  letterSpacing: 4,
                  shadows: [BoxShadow(color: verdictColor.withOpacity(0.5), blurRadius: 20)],
                ),
              ),
            ],
          ),
        ),
        const SizedBox(height: 24),
        
        const Text('// METRICS', style: TextStyle(color: AppTheme.primary, letterSpacing: 2, fontSize: 14)),
        const SizedBox(height: 12),
        _buildGlassCard(
          child: Column(
            children: [
              _buildMetricRow('Deception Probability', '${(_result!['probability'] * 100).toStringAsFixed(1)}%', verdictColor),
              const Divider(color: Colors.white10),
              _buildMetricRow('Confidence Score', '${(_result!['confidence'] * 100).toStringAsFixed(1)}%', AppTheme.primary),
            ],
          ),
        ),
        const SizedBox(height: 24),

        if (_result!['explanation'] != null && _result!['explanation']['top_indicators'] != null) ...[
          const Text('// KEY INDICATORS', style: TextStyle(color: AppTheme.primary, letterSpacing: 2, fontSize: 14)),
          const SizedBox(height: 12),
          ...(_result!['explanation']['top_indicators'] as List).take(4).map((indicator) {
            final isPositive = indicator['impact'] > 0;
            return Container(
              margin: const EdgeInsets.only(bottom: 12),
              child: _buildGlassCard(
                child: Row(
                  children: [
                    Icon(
                      isPositive ? Icons.keyboard_double_arrow_up : Icons.keyboard_double_arrow_down, 
                      color: isPositive ? AppTheme.deception : AppTheme.truth, 
                      size: 24
                    ),
                    const SizedBox(width: 16),
                    Expanded(
                      child: Text(
                        indicator['feature'].toString().replaceAll('_', ' ').toUpperCase(), 
                        style: const TextStyle(color: Colors.white, fontSize: 14, letterSpacing: 1)
                      )
                    ),
                    Text(
                      indicator['impact'].toStringAsFixed(3), 
                      style: TextStyle(color: AppTheme.textMuted, fontFamily: 'monospace', fontSize: 16)
                    ),
                  ],
                ),
              ),
            );
          }).toList(),
        ],
        
        const SizedBox(height: 32),
        ElevatedButton(
          onPressed: () => Navigator.pop(context),
          style: ElevatedButton.styleFrom(
            backgroundColor: verdictColor.withOpacity(0.1),
            foregroundColor: verdictColor,
            side: BorderSide(color: verdictColor.withOpacity(0.5)),
            minimumSize: const Size(double.infinity, 60)
          ),
          child: const Text('INITIATE NEW SCAN', style: TextStyle(letterSpacing: 2, fontWeight: FontWeight.bold)),
        ),
        const SizedBox(height: 24),
      ],
    );
  }
}
