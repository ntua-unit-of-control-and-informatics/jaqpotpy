def get_installed_packages() -> dict:
    """Retrieves a dictionary of installed packages and their versions using pip.freeze.

    Returns
    -------
        dict: A dictionary with package names as keys and versions as values.

    """
    try:
        from pip._internal.operations import freeze
    except ImportError:  # pip < 10.0
        from pip.operations import freeze

    pip_freeze_rows = freeze.freeze()
    packages_dict = {}
    for pkg in pip_freeze_rows:
        name, version = pkg.split("==")
        packages_dict[name] = version

    return packages_dict
