import 'package:flutter/material.dart';
import 'screens/camera_screen.dart';
import 'theme/app_theme.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const DeceptionApp());
}

class DeceptionApp extends StatelessWidget {
  const DeceptionApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Deception Analysis',
      debugShowCheckedModeBanner: false,
      theme: AppTheme.darkTheme,
      home: const CameraScreen(),
    );
  }
}
