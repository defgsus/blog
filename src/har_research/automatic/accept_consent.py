from capture import *


def accept_consent(cap: Capture):
    if accept_consent_button(cap):
        return True

    if accept_consent_iframe(cap):
        return True

    return False


def accept_consent_iframe(cap: Capture):
    iframes = []
    for iframe in cap.browser.find_elements_by_tag_name("iframe"):
        src = iframe.get_attribute("src")
        if "privacy" in src or "consent" in src:
            iframes.append(iframe)

    if not iframes:
        return False

    try:
        with cap.actions() as a:
            printe(f"click consent iframe {iframes[0]}")
            a.move_to_element(iframes[0])
            a.click()
            a.perform()
        cap.sleep_after_consent_click()
        return True
    except BaseException as e:
        printe(f"Error clicking consent iframe: {e}")


def accept_consent_button(cap: Capture):
    buttons = cap.browser.find_elements_by_tag_name("button")

    for text in ("accept", "agree", "consent", "zustimmen", "stimme zu"):
        for elem in buttons:
            if elem.text and text in elem.text.lower():
                try:
                    elem.click()
                    cap.sleep_after_consent_click()
                    return True
                except Exception as e:
                    print(f"Could not click accept button: {e}")

    return False
