import 'dart:async';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:permission_handler/permission_handler.dart';
import '../theme/app_theme.dart';
import 'analysis_screen.dart';

class CameraScreen extends StatefulWidget {
  const CameraScreen({super.key});

  @override
  State<CameraScreen> createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen> with SingleTickerProviderStateMixin {
  CameraController? _controller;
  List<CameraDescription> _cameras = [];
  bool _isRecording = false;
  int _recordDuration = 0;
  Timer? _timer;
  late AnimationController _pulseController;
  late Animation<double> _pulseAnimation;

  @override
  void initState() {
    super.initState();
    _pulseController = AnimationController(vsync: this, duration: const Duration(seconds: 1))..repeat(reverse: true);
    _pulseAnimation = Tween<double>(begin: 1.0, end: 1.2).animate(CurvedAnimation(parent: _pulseController, curve: Curves.easeInOut));
    _initializeCamera();
  }

  Future<void> _initializeCamera() async {
    Map<Permission, PermissionStatus> statuses = await [Permission.camera, Permission.microphone].request();
    if (statuses[Permission.camera]!.isGranted && statuses[Permission.microphone]!.isGranted) {
      _cameras = await availableCameras();
      if (_cameras.isNotEmpty) {
        // Find front camera first, fallback to first available
        CameraDescription frontCamera = _cameras.firstWhere(
            (c) => c.lensDirection == CameraLensDirection.front, 
            orElse: () => _cameras.first);
        _controller = CameraController(frontCamera, ResolutionPreset.high, enableAudio: true);
        await _controller!.initialize();
        if (mounted) setState(() {});
      }
    } else {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('Camera and Microphone permissions are required')));
      }
    }
  }

  void _startRecording() async {
    if (_controller == null || !_controller!.value.isInitialized || _isRecording) return;
    try {
      await _controller!.startVideoRecording();
      setState(() {
        _isRecording = true;
        _recordDuration = 0;
      });
      _timer = Timer.periodic(const Duration(seconds: 1), (timer) {
        setState(() => _recordDuration++);
      });
    } catch (e) {
      debugPrint('Error starting recording: $e');
    }
  }

  void _stopRecording() async {
    if (_controller == null || !_controller!.value.isRecordingVideo) return;
    try {
      _timer?.cancel();
      XFile videoFile = await _controller!.stopVideoRecording();
      setState(() => _isRecording = false);
      if (mounted) {
        Navigator.push(context, MaterialPageRoute(builder: (_) => AnalysisScreen(videoPath: videoFile.path)));
      }
    } catch (e) {
      debugPrint('Error stopping recording: $e');
    }
  }

  @override
  void dispose() {
    _controller?.dispose();
    _timer?.cancel();
    _pulseController.dispose();
    super.dispose();
  }

  String get _formattedTime {
    int minutes = _recordDuration ~/ 60;
    int seconds = _recordDuration % 60;
    return '${minutes.toString().padLeft(2, '0')}:${seconds.toString().padLeft(2, '0')}';
  }

  @override
  Widget build(BuildContext context) {
    if (_controller == null || !_controller!.value.isInitialized) {
      return const Scaffold(
        backgroundColor: AppTheme.background, 
        body: Center(child: CircularProgressIndicator(color: AppTheme.primary))
      );
    }

    return Scaffold(
      backgroundColor: Colors.black,
      body: Stack(
        children: [
          // Full screen camera preview
          Positioned.fill(
            child: FittedBox(
              fit: BoxFit.cover,
              child: SizedBox(
                width: _controller!.value.previewSize?.height ?? 1,
                height: _controller!.value.previewSize?.width ?? 1,
                child: CameraPreview(_controller!),
              ),
            ),
          ),
          
          // Cyberpunk Vignette/Gradient Overlay
          Positioned.fill(
            child: Container(
              decoration: BoxDecoration(
                gradient: LinearGradient(
                  begin: Alignment.topCenter,
                  end: Alignment.bottomCenter,
                  colors: [
                    AppTheme.background.withOpacity(0.8), 
                    Colors.transparent, 
                    Colors.transparent, 
                    AppTheme.background.withOpacity(0.9)
                  ],
                  stops: const [0.0, 0.15, 0.7, 1.0],
                ),
              ),
            ),
          ),

          // CyberHUD UI
          SafeArea(
            child: Column(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                // Top Bar
                Padding(
                  padding: const EdgeInsets.all(24.0),
                  child: Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text('DECEPTION', style: Theme.of(context).textTheme.titleLarge?.copyWith(color: AppTheme.primary, letterSpacing: 4.0)),
                          Text('ANALYSIS HUB', style: Theme.of(context).textTheme.bodySmall?.copyWith(color: AppTheme.textMuted, letterSpacing: 2.0)),
                        ],
                      ),
                      if (_isRecording)
                        Container(
                          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                          decoration: BoxDecoration(
                            color: Colors.black54,
                            borderRadius: BorderRadius.circular(20),
                            border: Border.all(color: AppTheme.deception.withOpacity(0.5)),
                          ),
                          child: Row(
                            children: [
                              ScaleTransition(
                                scale: _pulseAnimation, 
                                child: const Icon(Icons.circle, color: AppTheme.deception, size: 12)
                              ),
                              const SizedBox(width: 8),
                              Text(_formattedTime, style: const TextStyle(color: Colors.white, fontWeight: FontWeight.bold, fontSize: 16)),
                            ],
                          ),
                        )
                    ],
                  ),
                ),

                // Frame overlay targeting the face
                Expanded(
                  child: Center(
                    child: Container(
                      width: 250,
                      height: 350,
                      decoration: BoxDecoration(
                        border: Border.all(color: AppTheme.primary.withOpacity(0.3), width: 1),
                        borderRadius: BorderRadius.circular(20),
                      ),
                      child: Stack(
                        children: [
                          Positioned(top: 0, left: 0, child: _buildCorner(top: true, left: true)),
                          Positioned(top: 0, right: 0, child: _buildCorner(top: true, left: false)),
                          Positioned(bottom: 0, left: 0, child: _buildCorner(top: false, left: true)),
                          Positioned(bottom: 0, right: 0, child: _buildCorner(top: false, left: false)),
                        ],
                      ),
                    ),
                  ),
                ),

                // Record Button
                Padding(
                  padding: const EdgeInsets.only(bottom: 40.0),
                  child: GestureDetector(
                    onTap: _isRecording ? _stopRecording : _startRecording,
                    child: AnimatedContainer(
                      duration: const Duration(milliseconds: 300),
                      width: 80,
                      height: 80,
                      decoration: BoxDecoration(
                        shape: BoxShape.circle,
                        color: Colors.transparent,
                        border: Border.all(color: _isRecording ? AppTheme.deception : AppTheme.primary, width: 3),
                        boxShadow: [
                          BoxShadow(
                            color: (_isRecording ? AppTheme.deception : AppTheme.primary).withOpacity(0.3),
                            blurRadius: 20,
                            spreadRadius: 5
                          )
                        ]
                      ),
                      child: Center(
                        child: AnimatedContainer(
                          duration: const Duration(milliseconds: 300),
                          width: _isRecording ? 30 : 60,
                          height: _isRecording ? 30 : 60,
                          decoration: BoxDecoration(
                            color: _isRecording ? AppTheme.deception : AppTheme.primary,
                            borderRadius: BorderRadius.circular(_isRecording ? 8 : 30),
                          ),
                        ),
                      ),
                    ),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildCorner({required bool top, required bool left}) {
    return Container(
      width: 20,
      height: 20,
      decoration: BoxDecoration(
        border: Border(
          top: top ? const BorderSide(color: AppTheme.primary, width: 3) : BorderSide.none,
          bottom: !top ? const BorderSide(color: AppTheme.primary, width: 3) : BorderSide.none,
          left: left ? const BorderSide(color: AppTheme.primary, width: 3) : BorderSide.none,
          right: !left ? const BorderSide(color: AppTheme.primary, width: 3) : BorderSide.none,
        ),
      ),
    );
  }
}
