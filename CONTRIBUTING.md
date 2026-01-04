# Contributing to qmonsprt

Thank you for your interest in contributing to qmonsprt! This document provides guidelines and instructions for contributing.

## Code of Conduct

This project adheres to a code of conduct that all contributors are expected to follow. Please be respectful and constructive in all interactions.

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue using the [Bug Report template](.github/ISSUE_TEMPLATE/bug_report.md). Include:
- A clear description of the bug
- Steps to reproduce
- Expected vs. actual behavior
- Your environment details

### Suggesting Features

Feature suggestions are welcome! Please use the [Feature Request template](.github/ISSUE_TEMPLATE/feature_request.md) to describe:
- The feature and its motivation
- How it would be used
- Any alternatives you've considered

### Asking Questions

For questions about usage or implementation, use the [Question template](.github/ISSUE_TEMPLATE/question.md).

## Development Setup

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/your-username/qmonsprt.git
   cd qmonsprt
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -e ".[dev]"
   ```
4. Create a branch for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Coding Standards

- Follow PEP 8 style guidelines
- Write docstrings for all functions and classes
- Add unit tests for new functionality
- Ensure all tests pass: `pytest tests/`
- Update documentation as needed

## Testing

Before submitting, ensure:
- All existing tests pass: `pytest tests/`
- New tests are added for new functionality
- Code coverage is maintained or improved

## Submitting Changes

1. Commit your changes with clear, descriptive messages
2. Push to your fork
3. Open a Pull Request with:
   - A clear title and description
   - Reference to any related issues
   - Description of changes and testing performed

## Research Codebase

This is a research codebase supporting a published paper. When contributing:
- Maintain compatibility with existing analysis workflows
- Document any changes that affect numerical results
- Consider backward compatibility for saved data formats

## Questions?

Feel free to open an issue with the question template or contact the maintainers directly.

Thank you for contributing! 🎉

