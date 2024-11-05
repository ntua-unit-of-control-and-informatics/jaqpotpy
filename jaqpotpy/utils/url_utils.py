from urllib.parse import urlparse, urlunparse


def add_subdomain(url, subdomain):
    """:param url:
    :param subdomain:
    :return:
    """
    # Parse the original URL
    parsed_url = urlparse(url)

    # Split the hostname to add the subdomain
    hostname_parts = parsed_url.hostname.split(".")
    # Insert the new subdomain at the start
    hostname_parts.insert(0, subdomain)

    # Join the hostname back together
    new_hostname = ".".join(hostname_parts[1:])

    # Construct the new URL
    new_netloc = f"{subdomain}.{new_hostname}"
    new_url = urlunparse(parsed_url._replace(netloc=new_netloc))

    return new_url
