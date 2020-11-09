## Frequently Asked Questions

### Is there a way to get this field?

spotipy can only return fields documented on the Spotify web API https://developer.spotify.com/documentation/web-api/reference/

### How to use spotipy in an API?

Check out [this example Flask app](examples/app.py)

### Incorrect user

Error:

 - You get `You cannot create a playlist for another user`
 - You get `You cannot remove tracks from a playlist you don't own`

Solution:

 - Verify that you are signed in with the correct account on https://spotify.com
 - Remove your current token: `rm .cache-{userid}`
 - Request a new token by adding `show_dialog=True` to `spotipy.Spotify(auth_manager=SpotifyOAuth(show_dialog=True))`
 - Check that `spotipy.me()` shows the correct user id

### Why do I get 401 Unauthorized?

Error:

    spotipy.exceptions.SpotifyException: http status: 401, code:-1 - https://api.spotify.com/v1/
    Unauthorized.

Solution:

 - You are likely missing a scope when requesting the endpoint, check
https://developer.spotify.com/web-api/using-scopes/

### Search doesn't find some tracks

Problem: you can see a track on the Spotify app but searching for it using the API doesn't find it.
 
Solution: by default `search("abba")` works in the US market.
To search for in your current country, the [country indicator](https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2)
must be specified: `search("abba", market="DE")`.

### How do I obtain authorization in a headless/browserless environment?

If you cannot open a browser, set `open_browser=False` when instantiating SpotifyOAuth or SpotifyPKCE. You will be
prompted to open the authorization URI manually.  

See the [headless auth example](examples/headless.py).