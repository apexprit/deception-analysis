import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

class AppTheme {
  static const Color background = Color(0xFF0B0F19); // Deep dark slate
  static const Color surface = Color(0xFF151C2C);
  static const Color primary = Color(0xFF00F0FF); // Neon Cyan
  static const Color accent = Color(0xFF7000FF); // Deep Purple/Blue
  static const Color truth = Color(0xFF00FF9D); // Neon Emerald
  static const Color deception = Color(0xFFFF003C); // Neon Crimson
  static const Color text = Color(0xFFF8FAFC);
  static const Color textMuted = Color(0xFF94A3B8);

  static ThemeData get darkTheme {
    return ThemeData(
      brightness: Brightness.dark,
      scaffoldBackgroundColor: background,
      primaryColor: primary,
      colorScheme: const ColorScheme.dark(
        primary: primary,
        secondary: accent,
        surface: surface,
        error: deception,
      ),
      textTheme: GoogleFonts.interTextTheme(ThemeData.dark().textTheme).copyWith(
        displayLarge: GoogleFonts.spaceGrotesk(color: text, fontWeight: FontWeight.bold),
        titleLarge: GoogleFonts.spaceGrotesk(color: text, fontWeight: FontWeight.w600),
        bodyLarge: GoogleFonts.inter(color: text),
        bodyMedium: GoogleFonts.inter(color: textMuted),
      ),
      appBarTheme: AppBarTheme(
        backgroundColor: background,
        elevation: 0,
        centerTitle: true,
        titleTextStyle: GoogleFonts.spaceGrotesk(color: text, fontWeight: FontWeight.bold, fontSize: 18, letterSpacing: 2),
      ),
      elevatedButtonTheme: ElevatedButtonThemeData(
        style: ElevatedButton.styleFrom(
          backgroundColor: primary,
          foregroundColor: background,
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
          padding: const EdgeInsets.symmetric(vertical: 16, horizontal: 24),
        ),
      ),
    );
  }
}
