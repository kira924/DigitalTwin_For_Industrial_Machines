import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../screens/login_screen.dart';
import '../theme/dt_colors.dart';
import '../theme/dt_tokens.dart';
import 'routes.dart';
import 'state.dart';

class DigitalTwinApp extends StatelessWidget {
  const DigitalTwinApp({super.key});

  @override
  Widget build(BuildContext context) {
    final app = context.watch<AppState>();

    return MaterialApp(
      debugShowCheckedModeBanner: false,
      themeMode: app.themeMode,
      theme: _buildTheme(brightness: Brightness.light),
      darkTheme: _buildTheme(brightness: Brightness.dark),
      home: app.isBootstrapping
          ? const _BootScreen()
          : (app.isLoggedIn ? const AppShell() : const LoginScreen()),
    );
  }

  ThemeData _buildTheme({required Brightness brightness}) {
    final dark = brightness == Brightness.dark;
    final canvas = dark ? DTTokens.canvas : DTTokens.canvasLight;
    final surface = dark ? DTTokens.surface : DTTokens.surfaceLight;
    final border = dark ? DTTokens.borderSubtle : DTTokens.borderSubtleLight;
    final inputFill = dark ? DTTokens.inputFill : DTTokens.surfaceElevatedLight;
    final hintColor = dark ? DTTokens.textTertiary : DTTokens.textTertiaryLight;
    final textPrimary =
        dark ? DTTokens.textPrimary : DTTokens.textPrimaryLight;

    return ThemeData(
      brightness: brightness,
      useMaterial3: true,
      scaffoldBackgroundColor: canvas,
      colorScheme: dark
          ? const ColorScheme.dark(
              primary: DTTokens.accentPrimary,
              secondary: DTTokens.accentLive,
              surface: DTTokens.surface,
            )
          : const ColorScheme.light(
              primary: DTTokens.accentPrimary,
              secondary: DTTokens.accentLive,
              surface: DTTokens.surfaceLight,
            ),
      appBarTheme: const AppBarTheme(
        backgroundColor: Colors.transparent,
        elevation: 0,
        scrolledUnderElevation: 0,
        surfaceTintColor: Colors.transparent,
      ),
      cardTheme: CardThemeData(
        color: surface,
        elevation: 0,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(DTTokens.radiusMd),
          side: BorderSide.none,
        ),
      ),
      filledButtonTheme: FilledButtonThemeData(
        style: FilledButton.styleFrom(
          backgroundColor: DTTokens.accentPrimary,
          foregroundColor: Colors.white,
          textStyle: const TextStyle(
            fontWeight: FontWeight.w600,
            fontSize: 14,
            letterSpacing: 0.1,
          ),
          padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(DTTokens.radiusMd),
          ),
        ),
      ),
      inputDecorationTheme: InputDecorationTheme(
        filled: true,
        fillColor: inputFill,
        hintStyle: TextStyle(color: hintColor, fontSize: 14),
        contentPadding:
            const EdgeInsets.symmetric(horizontal: 14, vertical: 14),
        border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(DTTokens.radiusMd),
          borderSide: BorderSide(color: border),
        ),
        enabledBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(DTTokens.radiusMd),
          borderSide: BorderSide(color: border),
        ),
        focusedBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(DTTokens.radiusMd),
          borderSide: const BorderSide(color: DTTokens.accentLive, width: 1.4),
        ),
      ),
      sliderTheme: SliderThemeData(
        activeTrackColor: DTTokens.accentLive,
        inactiveTrackColor: border,
        thumbColor: DTTokens.accentLive,
        overlayColor: DTTokens.accentLive.alphaF(0.12),
        trackHeight: 3,
        thumbShape: const RoundSliderThumbShape(enabledThumbRadius: 7),
      ),
      switchTheme: SwitchThemeData(
        thumbColor: WidgetStateProperty.resolveWith((states) =>
            states.contains(WidgetState.selected)
                ? Colors.white
                : DTTokens.textTertiary),
        trackColor: WidgetStateProperty.resolveWith((states) =>
            states.contains(WidgetState.selected)
                ? DTTokens.accentPrimary
                : (dark
                    ? DTTokens.surfaceElevated
                    : DTTokens.surfaceElevatedLight)),
        trackOutlineColor: WidgetStateProperty.all(border),
      ),
      textTheme: TextTheme(
        bodyMedium: TextStyle(color: textPrimary, fontSize: 14),
      ),
    );
  }
}

class _BootScreen extends StatelessWidget {
  const _BootScreen();

  @override
  Widget build(BuildContext context) {
    final app = context.watch<AppState>();
    final palette = DTPalette.of(context);
    return Scaffold(
      backgroundColor: palette.canvas,
      body: Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Container(
              width: 64,
              height: 64,
              decoration: BoxDecoration(
                gradient: DT.grad,
                borderRadius: BorderRadius.circular(DTTokens.radiusLg),
              ),
              child: const Icon(
                Icons.factory_rounded,
                color: Colors.white,
                size: 30,
              ),
            ),
            const SizedBox(height: DTTokens.space20),
            const SizedBox(
              width: 22,
              height: 22,
              child: CircularProgressIndicator(
                strokeWidth: 2.4,
                color: DTTokens.accentLive,
              ),
            ),
            const SizedBox(height: DTTokens.space16),
            Text(
              'Initializing Smart Factory',
              style: DTTokens.h3(palette.textPrimary),
            ),
            const SizedBox(height: 6),
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: DTTokens.space24),
              child: Text(
                app.firebaseStatus,
                textAlign: TextAlign.center,
                style: DTTokens.caption(palette.textSecondary),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
