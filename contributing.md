# Contributing to dclm

Thank you for your interest in contributing to the dclm project! We welcome contributions of all kinds, including code, documentation, bug reports, and feature requests. This guide will help you get started and ensure your contributions are effectively integrated into the project.

## Table of Contents

- [Ways to Contribute](#ways-to-contribute)
- [Reporting Issues](#reporting-issues)
- [Finding an Issue to Work On](#finding-an-issue-to-work-on)
- [Submitting a Pull Request](#submitting-a-pull-request)
- [Coding Guidelines](#coding-guidelines)
- [Code of Conduct](#code-of-conduct)

## Ways to Contribute

There are many ways you can contribute to dclm:
- **Code Contributions**: Implement new features, fix bugs, or improve the existing codebase.
- **Documentation**: Enhance the documentation, correct typos, or add new sections.
- **Bug Reports and Feature Requests**: Help us identify bugs and suggest new features by reporting them.
- **Code Reviews and Discussions**: Review pull requests from other contributors and participate in discussions.

## Reporting Issues

If you encounter a bug or have a feature request, please report it by following these steps:
1. **Search for existing issues**: Before creating a new issue, search the existing issues to see if it has already been reported or suggested.
2. **Create a detailed report**: If you find no existing issue, [create a new one](https://github.com/mlfoundations/dclm/issues/new). 
3. Provide as much detail as possible to help us understand and reproduce the problem or consider the feature.

## Finding an Issue to Work On

If you are new to dclm or open-source development, we recommend starting with issues labeled "good first issue" or "docs". Here’s how to get started:
1. **Search for issues**: Check the "Issues" tab on our GitHub repository.
2. **Assign yourself an issue**: Comment on the issue to let others know you are working on it. If the issue is already assigned and inactive for a week, feel free to ask the current assignee if you can take over.

## Submitting a Pull Request

To contribute code to dclm, follow these steps:

1. **Fork the repository**: Create your own fork of the repository by clicking the "Fork" button on the GitHub page.
2. **Clone your fork**: Clone your forked repository to your local machine.
    ```bash
    git clone https://github.com/your-username/dclm.git
    cd dclm
    ```
3. **Create a feature branch**: Create a new branch for your feature or bug fix.
    ```bash
    git checkout -b my-feature-branch
    ```
4. **Make your changes**: Implement your feature or fix the bug.
5. **Commit your changes**: Commit your changes with a meaningful commit message.
    ```bash
    git add .
    git commit -m "Description of the feature or fix"
    ```
6. **Push to your fork**: Push your changes to your forked repository.
    ```bash
    git push origin my-feature-branch
    ```
7. **Open a pull request**: Go to the original repository and open a pull request. Provide a detailed description of your changes and link to any relevant issues.

## Coding Guidelines

To maintain consistency and quality in the codebase, please adhere to the following guidelines:
- **Style Guide**: Use the `black` linter to lint your code.
- **Testing**: Write tests for your code to ensure its correctness and prevent future regressions.
For comparison, see the existing `tests` module.
- **Documentation**: Update or add documentation as necessary to explain your changes.


## Code of Conduct

We abide by the principles of openness, respect, and consideration of others of the Python Software Foundation: https://www.python.org/psf/codeofconduct/.

Thank you for your contributions!