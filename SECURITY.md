# Security Policy

## Supported Versions

We release patches for security vulnerabilities for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 0.11.x  | :white_check_mark: |
| < 0.11  | :x:                |

## Reporting a Vulnerability

We take the security of SAHI seriously. If you believe you have found a security vulnerability in SAHI, please report it to us as described below.

### Where to Report

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, please report them via one of the following methods:

1. **GitHub Security Advisories** (Preferred)
   - Go to the [Security Advisories](https://github.com/obss/sahi/security/advisories) page
   - Click "Report a vulnerability"
   - Fill in the details of the vulnerability

2. **Email**
   - Send an email to the maintainers through GitHub
   - Include "SECURITY" in the subject line
   - Provide detailed information about the vulnerability

### What to Include

Please include the following information in your report:

- Type of vulnerability (e.g., remote code execution, information disclosure, etc.)
- Full paths of source file(s) related to the vulnerability
- The location of the affected source code (tag/branch/commit or direct URL)
- Any special configuration required to reproduce the issue
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact of the issue, including how an attacker might exploit it

### Response Timeline

- We will acknowledge your report within **3 business days**
- We will provide a detailed response within **7 business days** indicating the next steps
- We will keep you informed about the progress towards a fix and full announcement
- We may ask for additional information or guidance

### Disclosure Policy

- Security issues will be addressed with high priority
- Once a fix is available, we will:
  1. Release a patch version
  2. Publish a security advisory on GitHub
  3. Credit you (unless you prefer to remain anonymous)
  4. Update the CHANGELOG with security fix information

### Security Best Practices for Users

When using SAHI, we recommend:

1. **Keep SAHI updated** to the latest version
2. **Review dependencies** regularly for known vulnerabilities
3. **Validate inputs** when processing untrusted images or models
4. **Use virtual environments** to isolate SAHI and its dependencies
5. **Follow least privilege principle** when running SAHI in production
6. **Be cautious with model weights** from untrusted sources

### Known Security Considerations

- **Model Loading**: Be cautious when loading model weights from untrusted sources
- **Image Processing**: Validate and sanitize image inputs, especially from untrusted sources
- **File Operations**: SAHI performs file I/O operations; ensure proper permissions and path validation
- **Dependencies**: Some optional dependencies (PyTorch, TensorFlow, etc.) may have their own security considerations

### Security Updates

Security updates will be announced through:

- [GitHub Security Advisories](https://github.com/obss/sahi/security/advisories)
- [GitHub Releases](https://github.com/obss/sahi/releases)
- [CHANGELOG.md](./CHANGELOG.md)

### Additional Resources

- [GitHub Security Best Practices](https://docs.github.com/en/code-security)
- [OWASP Top Ten](https://owasp.org/www-project-top-ten/)
- [Python Security Best Practices](https://python.readthedocs.io/en/stable/library/security_warnings.html)

## Bug Bounty Program

Currently, we do not have a bug bounty program. However, we greatly appreciate security researchers who responsibly disclose vulnerabilities to us.

## Contact

For any security-related questions or concerns, please contact the maintainers through GitHub.

---

Thank you for helping keep SAHI and its users safe!
