import 'dart:io';
import 'package:dio/dio.dart';

class ApiService {
  // We will let the user change this IP in the UI later, or hardcode the AWS IP for now.
  // Make sure this matches the public IP of your AWS instance.
  static const String baseUrl = 'http://52.66.230.96:8000'; // AWS Public IP
  final Dio _dio = Dio(BaseOptions(
    baseUrl: baseUrl, 
    connectTimeout: const Duration(seconds: 15), 
    receiveTimeout: const Duration(seconds: 120)
  ));

  Future<Map<String, dynamic>> analyzeVideo(File videoFile, Function(int, int) onProgress) async {
    try {
      String fileName = videoFile.path.split('/').last;
      FormData formData = FormData.fromMap({
        "file": await MultipartFile.fromFile(videoFile.path, filename: fileName),
      });

      Response response = await _dio.post(
        '/analyze',
        data: formData,
        onSendProgress: onProgress,
      );

      return response.data;
    } on DioException catch (e) {
      throw Exception('Analysis failed: ${e.message}');
    } catch (e) {
      throw Exception('An error occurred: $e');
    }
  }
}
