import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../app/state.dart';
import '../theme/dt_colors.dart';
import '../theme/dt_tokens.dart';
import '../theme/dt_widgets.dart';

class LoginScreen extends StatefulWidget {
  const LoginScreen({super.key});

  @override
  State<LoginScreen> createState() => _LoginScreenState();
}

class _LoginScreenState extends State<LoginScreen> {
  final TextEditingController email =
      TextEditingController(text: 'belal19lol@gmail.com');
  final TextEditingController pass = TextEditingController(text: '123456');

  @override
  void dispose() {
    email.dispose();
    pass.dispose();
    super.dispose();
  }

  Future<void> _submit() async {
    final app = context.read<AppState>();
    final error = await app.login(email: email.text, password: pass.text);
    if (!mounted || error == null) return;

    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text(error)),
    );
  }

  void _openForgotPasswordSheet() {
    showModalBottomSheet<void>(
      context: context,
      isScrollControlled: true,
      backgroundColor: const Color(0xFF07111F),
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
      ),
      builder: (_) => _ForgotPasswordSheet(prefillEmail: email.text.trim()),
    );
  }

  @override
  Widget build(BuildContext context) {
    final app = context.watch<AppState>();
    final palette = DTPalette.of(context);
    return Scaffold(
      backgroundColor: palette.canvas,
      body: SafeArea(
        child: Center(
          child: SingleChildScrollView(
            padding: const EdgeInsets.all(DTTokens.space20),
            child: ConstrainedBox(
              constraints: const BoxConstraints(maxWidth: 440),
              child: Column(
                children: [
                  // Brand mark
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
                  const SizedBox(height: DTTokens.space16),
                  Text(
                    'Smart Factory Control',
                    style: DTTokens.h1(palette.textPrimary)
                        .copyWith(fontSize: 24),
                  ),
                  const SizedBox(height: 6),
                  Text(
                    'Real-time monitoring · Predictive maintenance',
                    style: DTTokens.body(palette.textSecondary),
                    textAlign: TextAlign.center,
                  ),
                  const SizedBox(height: DTTokens.space24),

                  // Cloud status banner
                  _CloudStatusBanner(
                    enabled: app.firebaseEnabled,
                    text: app.firebaseStatus,
                  ),
                  const SizedBox(height: DTTokens.space12),

                  // Sign-in form
                  DtCard(
                    padding: const EdgeInsets.all(DTTokens.space20),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          'Sign in',
                          style: DTTokens.h2(palette.textPrimary),
                        ),
                        const SizedBox(height: DTTokens.space16),
                        Text(
                          'Email',
                          style: DTTokens.label(palette.textSecondary),
                        ),
                        const SizedBox(height: 6),
                        TextField(
                          controller: email,
                          keyboardType: TextInputType.emailAddress,
                          decoration: const InputDecoration(
                            prefixIcon: Icon(Icons.mail_outline_rounded, size: 18),
                            hintText: 'name@factory.local',
                          ),
                        ),
                        const SizedBox(height: DTTokens.space12),
                        Row(
                          mainAxisAlignment: MainAxisAlignment.spaceBetween,
                          children: [
                            Text(
                              'Password',
                              style: DTTokens.label(palette.textSecondary),
                            ),
                            // ── Forgot password link ──────────────────────
                            GestureDetector(
                              onTap: _openForgotPasswordSheet,
                              child: Text(
                                'Forgot password?',
                                style: DTTokens.caption(DTTokens.accentPrimary)
                                    .copyWith(fontWeight: FontWeight.w600),
                              ),
                            ),
                          ],
                        ),
                        const SizedBox(height: 6),
                        TextField(
                          controller: pass,
                          obscureText: true,
                          decoration: const InputDecoration(
                            prefixIcon: Icon(Icons.lock_outline_rounded, size: 18),
                            hintText: '••••••',
                          ),
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(height: DTTokens.space16),

                  if (app.authBusy)
                    const Padding(
                      padding: EdgeInsets.only(bottom: DTTokens.space12),
                      child: SizedBox(
                        width: 22,
                        height: 22,
                        child: CircularProgressIndicator(
                          strokeWidth: 2.4,
                          color: DTTokens.accentLive,
                        ),
                      ),
                    ),

                  GradientButton(
                    text: app.authBusy ? 'Signing in…' : 'Enter Factory',
                    icon: Icons.arrow_forward_rounded,
                    onTap: app.authBusy ? () {} : _submit,
                  ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Forgot-password bottom sheet
// ─────────────────────────────────────────────────────────────────────────────

class _ForgotPasswordSheet extends StatefulWidget {
  const _ForgotPasswordSheet({required this.prefillEmail});
  final String prefillEmail;

  @override
  State<_ForgotPasswordSheet> createState() => _ForgotPasswordSheetState();
}

class _ForgotPasswordSheetState extends State<_ForgotPasswordSheet> {
  late final TextEditingController _email;

  bool _busy = false;
  bool _sent = false;
  String? _error;

  @override
  void initState() {
    super.initState();
    _email = TextEditingController(text: widget.prefillEmail);
  }

  @override
  void dispose() {
    _email.dispose();
    super.dispose();
  }

  Future<void> _send() async {
    final raw = _email.text.trim();
    if (!RegExp(r'^[^@\s]+@[^@\s]+\.[^@\s]+$').hasMatch(raw)) {
      setState(() => _error = 'Enter a valid email address.');
      return;
    }

    setState(() {
      _busy = true;
      _error = null;
    });

    final error = await context
        .read<AppState>()
        .resetPassword(email: raw);

    if (!mounted) return;
    setState(() {
      _busy = false;
      if (error == null) {
        _sent = true;
      } else {
        _error = error;
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    final palette = DTPalette.of(context);

    return Padding(
      padding: EdgeInsets.fromLTRB(
        20,
        16,
        20,
        20 + MediaQuery.of(context).viewInsets.bottom,
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Drag handle
          Center(
            child: Container(
              width: 36,
              height: 4,
              decoration: BoxDecoration(
                color: palette.borderStrong,
                borderRadius: BorderRadius.circular(DTTokens.radiusPill),
              ),
            ),
          ),
          const SizedBox(height: 16),

          // ── Success state ────────────────────────────────────────────────
          if (_sent) ...[
            Center(
              child: Container(
                width: 56,
                height: 56,
                decoration: BoxDecoration(
                  color: DTTokens.statusHealthy.alphaF(0.12),
                  shape: BoxShape.circle,
                ),
                child: const Icon(
                  Icons.mark_email_read_outlined,
                  color: DTTokens.statusHealthy,
                  size: 26,
                ),
              ),
            ),
            const SizedBox(height: 16),
            Text(
              'Reset link sent',
              style: DTTokens.h2(palette.textPrimary),
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 8),
            Text(
              'Check your inbox at ${_email.text.trim()}. '
              'Follow the link to set a new password. '
              'If you don\'t see it, check your spam folder.',
              style: DTTokens.body(palette.textSecondary),
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 24),
            SizedBox(
              width: double.infinity,
              child: FilledButton(
                onPressed: () => Navigator.of(context).pop(),
                child: const Text('Done'),
              ),
            ),
          ]

          // ── Form state ───────────────────────────────────────────────────
          else ...[
            Text(
              'Reset password',
              style: DTTokens.h2(palette.textPrimary),
            ),
            const SizedBox(height: 6),
            Text(
              'Enter your account email. We\'ll send a reset link so you '
              'can set a new password.',
              style: DTTokens.body(palette.textSecondary),
            ),
            const SizedBox(height: 20),
            Text(
              'Email',
              style: DTTokens.label(palette.textSecondary),
            ),
            const SizedBox(height: 6),
            TextField(
              controller: _email,
              enabled: !_busy,
              autofocus: widget.prefillEmail.isEmpty,
              keyboardType: TextInputType.emailAddress,
              decoration: InputDecoration(
                prefixIcon:
                    const Icon(Icons.mail_outline_rounded, size: 18),
                hintText: 'name@factory.local',
                errorText: _error,
              ),
              onSubmitted: (_) => _send(),
            ),
            const SizedBox(height: 20),
            if (_busy)
              const Center(child: CircularProgressIndicator())
            else
              SizedBox(
                width: double.infinity,
                child: FilledButton.icon(
                  onPressed: _send,
                  icon: const Icon(Icons.send_rounded, size: 16),
                  label: const Text('Send reset link'),
                ),
              ),
            const SizedBox(height: 8),
            SizedBox(
              width: double.infinity,
              child: TextButton(
                onPressed: () => Navigator.of(context).pop(),
                child: const Text('Cancel'),
              ),
            ),
          ],
        ],
      ),
    );
  }
}

class _CloudStatusBanner extends StatelessWidget {
  const _CloudStatusBanner({required this.enabled, required this.text});

  final bool enabled;
  final String text;

  @override
  Widget build(BuildContext context) {
    final color = enabled ? DTTokens.statusHealthy : DTTokens.statusWarning;
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
      decoration: BoxDecoration(
        color: color.alphaF(0.08),
        borderRadius: BorderRadius.circular(DTTokens.radiusMd),
        border: Border.all(color: color.alphaF(0.25)),
      ),
      child: Row(
        children: [
          Icon(
            enabled ? Icons.cloud_done_rounded : Icons.cloud_off_rounded,
            color: color,
            size: 16,
          ),
          const SizedBox(width: 10),
          Expanded(
            child: Text(
              text,
              style: TextStyle(
                color: color,
                fontSize: 12,
                fontWeight: FontWeight.w600,
                height: 1.3,
              ),
            ),
          ),
        ],
      ),
    );
  }
}
