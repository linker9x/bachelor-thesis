from datetime import datetime


def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        total_sec= (datetime.now() - start_time).total_seconds()
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)

    # return 'formatted:' + str(thour) + ' h ' + str(tmin) + ' m ' \
    #        + str(round(tsec, 2)) + ' s ; in seconds: ' + str(round(total_sec, 2))

    return str(round(total_sec, 4))

