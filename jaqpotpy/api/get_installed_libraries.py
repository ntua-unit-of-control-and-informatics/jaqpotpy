import subprocess
from jaqpotpy.api.openapi.models.library import Library


def get_installed_libraries():
    """
    Retrieve a list of installed Python libraries and their versions.

    This function runs the `pip freeze` command to get a list of installed
    Python packages and their versions. It then parses the output and
    creates a list of Library objects.

    Returns:
        list: A list of Library objects, each containing the name and version
              of an installed Python package.
    """
    # Run the pip freeze command
    result = subprocess.run(["pip", "freeze"], stdout=subprocess.PIPE, text=True)

    # Split the output into lines
    packages = result.stdout.splitlines()

    # Create a list of dictionaries for each package
    packages_list = []
    for package in packages:
        if "==" in package:
            name, version = package.split("==")
            packages_list.append({"name": name, "version": version})
        elif " @ " in package:
            name, _ = package.split(" @ ")
            version = "Unknown"
            packages_list.append({"name": name, "version": version})

    libraries_list = [
        Library(name=library["name"], version=library["version"])
        for library in packages_list
    ]

    return libraries_list
