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
  // The Autocomplete widget owns its field controller; we store a reference
  // so _submit() and the forgot-password sheet can read the typed value.
  TextEditingController? _emailCtrl;
  final TextEditingController pass = TextEditingController();
  final FocusNode _passwordFocus = FocusNode();
  bool _passwordVisible = false;

  @override
  void dispose() {
    // _emailCtrl is owned by Autocomplete — do not dispose it here.
    pass.dispose();
    _passwordFocus.dispose();
    super.dispose();
  }

  Future<void> _submit() async {
    final app = context.read<AppState>();
    final error = await app.login(
      email: _emailCtrl?.text.trim() ?? '',
      password: pass.text,
    );
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
      builder: (_) =>
          _ForgotPasswordSheet(prefillEmail: _emailCtrl?.text.trim() ?? ''),
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
                        Autocomplete<String>(
                          optionsBuilder: (TextEditingValue value) {
                            const suggestion = 'Belal19lol@gmail.com';
                            if (value.text.isEmpty) return const [suggestion];
                            if (suggestion
                                .toLowerCase()
                                .contains(value.text.toLowerCase())) {
                              return const [suggestion];
                            }
                            return const [];
                          },
                          onSelected: (String selection) {
                            _emailCtrl?.text = selection;
                            FocusScope.of(context)
                                .requestFocus(_passwordFocus);
                          },
                          fieldViewBuilder: (
                            BuildContext ctx,
                            TextEditingController fieldCtrl,
                            FocusNode fieldFocus,
                            VoidCallback onFieldSubmitted,
                          ) {
                            _emailCtrl = fieldCtrl;
                            return TextField(
                              controller: fieldCtrl,
                              focusNode: fieldFocus,
                              keyboardType: TextInputType.emailAddress,
                              textInputAction: TextInputAction.next,
                              enabled: !app.authBusy,
                              onSubmitted: (_) => FocusScope.of(context)
                                  .requestFocus(_passwordFocus),
                              decoration: const InputDecoration(
                                prefixIcon: Icon(
                                    Icons.mail_outline_rounded,
                                    size: 18),
                                hintText: 'name@factory.local',
                              ),
                            );
                          },
                          optionsViewBuilder: (
                            BuildContext ctx,
                            AutocompleteOnSelected<String> onSelected,
                            Iterable<String> options,
                          ) {
                            return Align(
                              alignment: Alignment.topLeft,
                              child: Material(
                                color: Colors.transparent,
                                child: Container(
                                  margin: const EdgeInsets.only(top: 4),
                                  decoration: BoxDecoration(
                                    color: const Color(0xFF0E1A2A),
                                    borderRadius:
                                        BorderRadius.circular(DTTokens.radiusMd),
                                    border: Border.all(
                                      color: Colors.white.alphaF(0.08),
                                    ),
                                  ),
                                  child: Column(
                                    mainAxisSize: MainAxisSize.min,
                                    children: options.map((option) {
                                      return InkWell(
                                        onTap: () => onSelected(option),
                                        borderRadius: BorderRadius.circular(
                                            DTTokens.radiusMd),
                                        child: Padding(
                                          padding: const EdgeInsets.symmetric(
                                            horizontal: 14,
                                            vertical: 12,
                                          ),
                                          child: Row(
                                            children: [
                                              const Icon(
                                                Icons
                                                    .person_outline_rounded,
                                                size: 16,
                                                color:
                                                    DTTokens.accentPrimary,
                                              ),
                                              const SizedBox(width: 10),
                                              Text(
                                                option,
                                                style: TextStyle(
                                                  color: palette.textPrimary,
                                                  fontSize: 14,
                                                ),
                                              ),
                                            ],
                                          ),
                                        ),
                                      );
                                    }).toList(),
                                  ),
                                ),
                              ),
                            );
                          },
                        ),
                        const SizedBox(height: DTTokens.space12),
                        Row(
                          mainAxisAlignment: MainAxisAlignment.spaceBetween,
                          children: [
                            Text(
                              'Password',
                              style: DTTokens.label(palette.textSecondary),
                            ),
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
                          focusNode: _passwordFocus,
                          obscureText: !_passwordVisible,
                          textInputAction: TextInputAction.done,
                          enabled: !app.authBusy,
                          onSubmitted: (_) => _submit(),
                          decoration: InputDecoration(
                            prefixIcon: const Icon(Icons.lock_outline_rounded, size: 18),
                            hintText: '••••••',
                            suffixIcon: IconButton(
                              icon: Icon(
                                _passwordVisible
                                    ? Icons.visibility_off_outlined
                                    : Icons.visibility_outlined,
                                size: 18,
                              ),
                              onPressed: () =>
                                  setState(() => _passwordVisible = !_passwordVisible),
                              tooltip: _passwordVisible ? 'Hide password' : 'Show password',
                            ),
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
