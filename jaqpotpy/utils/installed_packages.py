def get_installed_packages():

    try: 
        from pip._internal.operations import freeze
    except ImportError: # pip < 10.0
        from pip.operations import freeze

    pip_freeze_rows = freeze.freeze()
    packages_dict = {}
    for pkg in pip_freeze_rows:
        name, version = pkg.split('==')
        packages_dict[name] = version
    
    return packages_dict
    
