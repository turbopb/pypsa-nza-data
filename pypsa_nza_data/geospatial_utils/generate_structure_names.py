#   Here's a Python function that takes two names in the format `
#   "INV-MAN-A-0001-Structure"` and `"INV-MAN-A-0355-Structure"` and returns a 
#   list of structured names incrementing the numeric portion (`n`) from `0001` 
#   to `0355`:

def generate_structure_names(start_desc, end_desc):
    """
    Generate a list of structure names from start_desc to end_desc (inclusive).

    Args:
        start_desc (str): e.g., "INV-MAN-A-0001-Structure"
        end_desc (str): e.g., "INV-MAN-A-0355-Structure"

    Returns:
        list of str: structure names from start to end.
    """
    try:
        # Extract prefix and numeric part
        start_prefix, start_number = start_desc.rsplit('-', 2)[0], start_desc.rsplit('-', 2)[1]
        end_prefix, end_number = end_desc.rsplit('-', 2)[0], end_desc.rsplit('-', 2)[1]

        # Ensure the prefixes match
        if start_prefix != end_prefix:
            raise ValueError("Start and end descriptions must have the same prefix")

        # Parse numbers
        start_num = int(start_number)
        end_num = int(end_number)

        if start_num > end_num:
            raise ValueError("Start number must be less than or equal to end number")

        # Rebuild full structure names
        structure_names = [
            f"{start_prefix}-{str(n).zfill(4)}-Structure" for n in range(start_num, end_num + 1)
        ]

        return structure_names

    except Exception as e:
        print(f"Error: {e}")
        return []

# Example Usage

if __name__== "__main__":

    names = generate_structure_names("INV-MAN-A-0001-Structure", "INV-MAN-A-0010-Structure")
    print(names[:10])    # First 10
    #print(names[-10:])   # Last 10
    print(len(names))   # Should print 10
    
    
    ### ? Output Preview``
    # ['INV-MAN-A-0001-Structure', 'INV-MAN-A-0002-Structure', 'INV-MAN-A-0003-Structure']
    # ['INV-MAN-A-0353-Structure', 'INV-MAN-A-0354-Structure', 'INV-MAN-A-0355-Structure']
    # #355

#Let me know if you’d like it to support partial numeric suffixes (e.g., skip '0355A') or do prefix grouping.
